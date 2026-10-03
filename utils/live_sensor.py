"""Optional live traffic sensor: turn observed packets into scored flow records.

The sensor is a defensive monitoring tool that inspects traffic on interfaces
**you explicitly choose on your own machine**. It only ever reads flow metadata
(the 5-tuple plus counters and timings); payloads are never assembled, parsed or
stored. Anything captured can be scored with the trained model, printed as
alerts, written to CSV, or pushed into the running web dashboard.

Legal note: capture traffic only on systems and networks you are authorised to
monitor.

Usage::

    python utils/live_sensor.py --list-interfaces
    python utils/live_sensor.py --iface Wi-Fi --seconds 60
    python utils/live_sensor.py --iface eth0 --csv captures/flows.csv
    python utils/live_sensor.py --iface eth0 --api http://127.0.0.1:5000/api/sensor/sample

Requirements: ``scapy`` plus a packet-capture driver. On Windows install
Npcap (https://npcap.com) in WinPcap API-compatible mode and run the terminal as
Administrator; on Linux run as root or grant CAP_NET_RAW.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd  # noqa: E402

from training import preprocess as pp  # noqa: E402
from utils.prediction import PREDICTION_MALICIOUS, ModelNotAvailableError, get_predictor  # noqa: E402

PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}
PROTOCOL_NUMBERS = {name: number for number, name in PROTOCOL_NAMES.items()}

TCP_FLAG_LETTERS = ((0x01, "F"), (0x02, "S"), (0x04, "R"), (0x08, "P"),
                    (0x10, "A"), (0x20, "U"), (0x40, "E"), (0x80, "C"))

DEFAULT_FLOW_TIMEOUT = 20.0
DEFAULT_MAX_FLOWS = 20000


class CaptureUnavailable(Exception):
    """Raised when the operating system refuses packet capture."""


@dataclass
class Flow:
    """Bidirectional statistics for a single 5-tuple."""

    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    protocol_type: int
    first_seen: float
    last_seen: float
    forward_packets: int = 0
    forward_bytes: int = 0
    reverse_packets: int = 0
    reverse_bytes: int = 0
    forward_flags: set = field(default_factory=set)
    reverse_flags: set = field(default_factory=set)
    inter_arrivals: List[float] = field(default_factory=list)

    @property
    def packets(self) -> int:
        return self.forward_packets + self.reverse_packets

    @property
    def bytes(self) -> int:
        return self.forward_bytes + self.reverse_bytes

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 0.0)

    def record(self) -> Dict[str, Any]:
        """Return the record in the canonical schema used by the model."""
        packets = max(self.packets, 1)
        intervals = [value for value in self.inter_arrivals if value > 0]
        mean_iat = sum(intervals) / len(intervals) if intervals else 0.0
        direction = (
            pp.DIRECTION_UNIDIRECTIONAL
            if self.reverse_packets == 0 or self.forward_packets == 0
            else (pp.DIRECTION_FORWARD if self.forward_packets >= self.reverse_packets else pp.DIRECTION_REVERSE)
        )
        return {
            "Flow_ID": f"{self.source_ip}:{self.source_port}-{self.destination_ip}:{self.destination_port}",
            "Source_IP": self.source_ip,
            "Destination_IP": self.destination_ip,
            "Source_Port": int(self.source_port),
            "Destination_Port": int(self.destination_port),
            "Protocol": self.protocol,
            "Protocol_Type": int(self.protocol_type),
            "Packet_Count": int(self.packets),
            "Byte_Count": int(self.bytes),
            "Packet_Length": round(self.bytes / packets, 2),
            "Flow_Duration": int(self.duration * 1_000_000),
            "Forward_Packets": int(self.forward_packets),
            "Forward_Bytes": int(self.forward_bytes),
            "Reverse_Packets": int(self.reverse_packets),
            "Reverse_Bytes": int(self.reverse_bytes),
            "TCP_Flags": ("".join(sorted(self.forward_flags)) or "") if self.protocol == "TCP" else "",
            "Inter_Arrival_Time": round(mean_iat, 6),
            "Traffic_Direction": direction,
        }


class FlowAggregator:
    """Aggregate packets into bidirectional flows and expire idle flows.

    The class is deliberately free of any capture dependency so it can be unit
    tested with synthetic packets.
    """

    def __init__(self, flow_timeout: float = DEFAULT_FLOW_TIMEOUT,
                 max_flows: int = DEFAULT_MAX_FLOWS) -> None:
        self.flow_timeout = float(flow_timeout)
        self.max_flows = int(max_flows)
        self._flows: "OrderedDict[tuple, Flow]" = OrderedDict()
        self.packets_seen = 0
        self.packets_ignored = 0

    def add_packet(self, source_ip: str, destination_ip: str, source_port: int, destination_port: int,
                   protocol_type: int, length: int, timestamp: Optional[float] = None,
                   flags: int = 0) -> None:
        """Register one observed packet and update its flow statistics."""
        now = time.time() if timestamp is None else float(timestamp)
        protocol = PROTOCOL_NAMES.get(int(protocol_type), str(protocol_type))
        key = (source_ip, destination_ip, int(source_port), int(destination_port), int(protocol_type))

        flow = self._flows.get(key)
        if flow is None:
            reverse_key = (destination_ip, source_ip, int(destination_port), int(source_port), int(protocol_type))
            reverse = self._flows.get(reverse_key)
            if reverse is not None:
                self._apply(reverse, source_ip, destination_ip, now, int(length), flags, reverse_direction=True)
                reverse.last_seen = max(reverse.last_seen, now)
                self.packets_seen += 1
                self._flows.move_to_end(reverse_key)
                return
            flow = Flow(
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=int(source_port),
                destination_port=int(destination_port),
                protocol=protocol,
                protocol_type=int(protocol_type),
                first_seen=now,
                last_seen=now,
            )
            self._flows[key] = flow
            while len(self._flows) > self.max_flows:
                self._flows.popitem(last=False)

        self._apply(flow, source_ip, destination_ip, now, int(length), flags,
                    reverse_direction=False)
        flow.last_seen = max(flow.last_seen, now)
        self.packets_seen += 1
        self._flows.move_to_end(key)

    def _apply(self, flow: Flow, source_ip: str, destination_ip: str, now: float,
               length: int, flags: int, reverse_direction: bool) -> None:
        previous = flow.last_seen
        gap = max(now - previous, 0.0)
        if 0 < gap < 60:
            flow.inter_arrivals.append(gap)
        letters = {letter for bit, letter in TCP_FLAG_LETTERS if flags & bit}
        if reverse_direction:
            flow.reverse_packets += 1
            flow.reverse_bytes += max(length, 0)
            flow.reverse_flags |= letters
        else:
            flow.forward_packets += 1
            flow.forward_bytes += max(length, 0)
            flow.forward_flags |= letters

    def expire(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Remove and return flows that have been idle for longer than the timeout."""
        moment = time.time() if now is None else float(now)
        completed: List[Dict[str, Any]] = []
        for key in list(self._flows.keys()):
            flow = self._flows[key]
            if moment - flow.last_seen >= self.flow_timeout:
                completed.append(flow.record())
                del self._flows[key]
        return completed

    def drain(self) -> List[Dict[str, Any]]:
        """Return and clear every flow currently held."""
        completed = [flow.record() for flow in self._flows.values()]
        self._flows.clear()
        return completed

    def active_count(self) -> int:
        return len(self._flows)

    def records(self) -> List[Dict[str, Any]]:
        return [flow.record() for flow in self._flows.values()]


# ------------------------------------------------------------------ capture layer


def _npcap_guid_map() -> Dict[str, str]:
    """Map Npcap interface GUIDs (``NPF_{...}``) to their Windows adapter names.

    ``Get-NetAdapter`` is preferred because it reports the GUID next to the
    friendly name; the registry is used as a fallback when the cmdlet is
    unavailable (older Windows, restricted PowerShell).
    """
    mapping: Dict[str, str] = {}
    if os.name != "nt":
        return mapping

    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-NetAdapter | ForEach-Object { \"$($_.InterfaceGuid)|$($_.Name)\" }"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in output.splitlines():
            if "|" not in line:
                continue
            guid, _, name = line.partition("|")
            guid = guid.strip().strip("{}").lower()
            name = name.strip()
            if guid and name:
                mapping[guid] = name
    except Exception:  # noqa: BLE001 - fall through to the registry
        mapping = {}

    if mapping:
        return mapping

    import winreg  # noqa: PLC0415

    base = r"SYSTEM\CurrentControlSet\Control\Network\{4D36E972-E325-11CE-BFC1-08002BE10318}"
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    with key:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(key, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(key, sub + r"\Connection") as connection:
                    name, _ = winreg.QueryValueEx(connection, "Name")
                    mapping[sub.strip("{}").lower()] = str(name)
            except OSError:
                continue
    return mapping


def _npcap_key(raw_name: str) -> str:
    """Extract the bare GUID from a scapy interface name such as ``NPF_{GUID}``."""
    text = raw_name.split("\\")[-1].strip().rstrip("}")
    if text.upper().startswith("NPF_"):
        text = text[4:]
    return text.strip("{}").lower()


def _windows_adapters() -> List[Dict[str, str]]:
    """Return Windows adapters with their name, GUID and connection state."""
    adapters: List[Dict[str, str]] = []
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-NetAdapter | ForEach-Object { "
             "\"$($_.InterfaceGuid)|$($_.Name)|$($_.Status)\" }"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return adapters
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        guid, name, status = (part.strip() for part in parts[:3])
        if not name:
            continue
        adapters.append({
            "name": name,
            "guid": guid.strip("{}").lower(),
            "status": status,
            "os": "windows",
        })
    return adapters


def list_interfaces() -> List[Dict[str, str]]:
    """Return the interfaces the operating system reports as usable."""
    interfaces: List[Dict[str, str]] = []

    if os.name == "nt":
        adapters = _windows_adapters()
        guid_to_scapy: Dict[str, str] = {}
        for adapter in adapters:
            interfaces.append({
                "name": adapter["name"],
                "os": "windows",
                "scapy_name": f"NPF_{adapter['guid']}",
                "status": adapter["status"],
            })
            guid_to_scapy[adapter["guid"]] = f"NPF_{adapter['guid']}"
    else:
        try:
            output = subprocess.run(["ip", "-o", "link", "show"], capture_output=True,
                                    text=True, timeout=20).stdout
            for line in output.splitlines():
                if ":" not in line:
                    continue
                name = line.split(":")[1].strip().split("@")[0]
                if name:
                    interfaces.append({"name": name, "os": "linux"})
        except Exception:  # noqa: BLE001
            pass
        guid_to_scapy = {}

    try:
        from scapy.all import get_if_list  # noqa: PLC0415

        known = {item["name"] for item in interfaces}
        for raw in get_if_list():
            if os.name != "nt":
                if raw in known:
                    continue
                display = raw
            else:
                guid = _npcap_key(raw)
                display = guid_to_scapy.get(guid, "").replace("NPF_", "")
                if not display:
                    # GUID with no matching adapter: keep the raw form, it is still valid
                    display = raw
            if display and display not in known:
                known.add(display)
                interfaces.append({"name": display, "os": "raw-socket", "scapy_name": raw})
    except Exception:  # noqa: BLE001 - scapy optional
        pass
    return interfaces


def resolve_interface(name: str) -> str:
    """Translate a friendly adapter name into the identifier scapy expects.

    On Windows scapy reports Npcap GUIDs, so a user-supplied name such as
    ``Wi-Fi`` or ``Ethernet 3`` is matched back to the raw ``NPF_{...}`` string.
    """
    if os.name != "nt":
        return name
    if name.upper().startswith("NPF_") or name.upper().startswith("NPF{"):
        return name
    wanted = name.strip().lower()
    for item in list_interfaces():
        if item["name"].lower() == wanted:
            return item.get("scapy_name", item["name"])
    return name
    if name.upper().startswith("NPF_") or name.upper().startswith("NPF{"):
        return name
    wanted = name.strip().lower()
    for item in list_interfaces():
        if item["name"].lower() == wanted:
            return item.get("scapy_name", item["name"])
    return name


def is_elevated() -> bool:
    """Return ``True`` when the process can open a raw capture socket."""
    if os.name != "nt":
        return os.geteuid() == 0
    try:
        import ctypes  # noqa: PLC0415

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001 - non-fatal, assume not elevated
        return False


def npcap_installed() -> bool:
    """Detect the Npcap/WinPcap driver without importing a capture library."""
    if os.name != "nt":
        return True
    for name in ("Npcap", "wpcap"):
        if os.path.exists(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", name + ".dll")):
            return True
    return False


def scapy_installed() -> bool:
    try:
        import scapy  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def check_prerequisites() -> List[Dict[str, Any]]:
    """Return the status of every live-capture requirement, in setup order."""
    checks = [
        {"name": "scapy installed", "ok": scapy_installed(),
         "fix": "pip install scapy"},
    ]
    if os.name == "nt":
        checks.append({
            "name": "Npcap driver installed", "ok": npcap_installed(),
            "fix": "install Npcap from https://npcap.com and keep "
                   "'WinPcap API-compatible mode' enabled",
        })
        checks.append({
            "name": "terminal running as Administrator", "ok": is_elevated(),
            "fix": "close the terminal and re-open it with 'Run as administrator'",
        })
    else:
        checks.append({
            "name": "CAP_NET_RAW (root or capability)", "ok": is_elevated(),
            "fix": "run with sudo, or grant the capability: "
                   "sudo setcap cap_net_raw,cap_net_admin=eip $(which python)",
        })
    return checks


def _capture_guidance() -> str:
    """Explain exactly which prerequisite is unmet."""
    missing = [check for check in check_prerequisites() if not check["ok"]]
    if not missing:
        return "capture prerequisites are satisfied"
    lines = [f"Live capture is unavailable ({len(missing)} unmet requirement"
             f"{'s' if len(missing) > 1 else ''}):", ""]
    lines.extend(f"  [ ] {check['name']}\n      -> {check['fix']}" for check in missing)
    if os.name == "nt":
        lines.append("\n  3. Restart Windows if the Npcap installer asks for it.")
    return "\n".join(lines)


def _extract(packet) -> Optional[Dict[str, Any]]:
    """Pull 5-tuple metadata out of a scapy packet (payload is never touched)."""
    ip_layer = packet.getlayer("IP") or packet.getlayer("IPv6")
    if ip_layer is None:
        return None
    source = ip_layer.src
    destination = ip_layer.dst
    source_port = destination_port = 0
    flags = 0
    for layer_name in ("TCP", "UDP"):
        layer = packet.getlayer(layer_name)
        if layer is not None:
            source_port = int(getattr(layer, "sport", 0) or 0)
            destination_port = int(getattr(layer, "dport", 0) or 0)
            flags = int(getattr(layer, "flags", 0) or 0)
            break
    protocol_type = int(ip_layer.proto if hasattr(ip_layer, "proto") else 6)
    return {
        "source_ip": source,
        "destination_ip": destination,
        "source_port": source_port,
        "destination_port": destination_port,
        "protocol_type": protocol_type,
        "length": int(len(packet)),
        "flags": flags,
    }


class FlowSensor:
    """Capture packets on one interface and emit scored flow records."""

    def __init__(self, interface: str, flow_timeout: float = DEFAULT_FLOW_TIMEOUT,
                 verbose: bool = False) -> None:
        self.interface = interface
        self.aggregator = FlowAggregator(flow_timeout=flow_timeout)
        self.verbose = verbose
        self._sniffer = None
        self.sensor_error: Optional[str] = None
        self._scapy = None

    def _load_scapy(self):
        if self._scapy is not None:
            return self._scapy
        try:
            from scapy.all import AsyncSniffer, conf  # noqa: PLC0415

            self._scapy = (AsyncSniffer, conf)
        except Exception as exc:  # noqa: BLE001
            raise CaptureUnavailable(
                "scapy is not installed. Run 'pip install scapy' to enable the live sensor."
            ) from exc
        return self._scapy

    def start(self) -> None:
        """Begin capturing in a background thread."""
        try:
            sniffer_class, conf = self._load_scapy()
        except CaptureUnavailable as exc:
            self.sensor_error = str(exc)
            raise

        def handler(packet) -> None:
            fields = _extract(packet)
            if fields is None:
                self.aggregator.packets_ignored += 1
                return
            self.aggregator.add_packet(**fields)

        conf.use_pcap = False
        try:
            self._sniffer = sniffer_class(iface=resolve_interface(self.interface), store=False, prn=handler)
            self._sniffer.start()
            self._sniffer.join(2.0)
            if not getattr(self._sniffer, "running", False):
                raise CaptureUnavailable(_capture_guidance())
        except CaptureUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            self.sensor_error = _capture_guidance()
            raise CaptureUnavailable(self.sensor_error) from exc

    def stop(self) -> None:
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._sniffer = None

    def running(self) -> bool:
        return bool(self._sniffer is not None and getattr(self._sniffer, "running", False))

    def collect(self) -> List[Dict[str, Any]]:
        """Expired flows first, then any still-open flows when asked to flush."""
        return self.aggregator.expire()


# ------------------------------------------------------------------- scoring layer


def score_records(records: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Score live flow records with the trained model and return the results table."""
    if not records:
        return None
    try:
        predictor = get_predictor()
    except Exception:  # noqa: BLE001
        return None
    if not predictor.available:
        return None

    frame = pp.canonicalize_columns(pd.DataFrame(records))[0]
    for column in ("Source_IP", "Destination_IP"):
        if column not in frame.columns:
            frame[column] = "unknown"
    if "Source_Port" not in frame.columns:
        frame["Source_Port"] = 0
    if "Destination_Port" not in frame.columns:
        frame["Destination_Port"] = 0
    label_column = pp.find_label_column(frame.columns)
    frame = pp.clean_frame(frame, label_column)
    frame["Label_normalised"] = "unlabelled"
    frame = pp.add_engineered_features(frame)
    result = predictor.predict_frame(frame)
    return result.records_frame


def push_to_api(url: str, records: List[Dict[str, Any]], timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    """Send flow records to the dashboard's sensor endpoint."""
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(
        url, data=json.dumps({"records": records}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not push to {url}: {exc}", file=sys.stderr)
        return None


def _print_records(records: List[Dict[str, Any]], scored: Optional[pd.DataFrame]) -> None:
    if scored is None:
        print("no scored data (model unavailable)")
        return
    for _, row in scored.iterrows():
        verdict = row["Prediction"]
        marker = "!!" if verdict == PREDICTION_MALICIOUS else "  "
        print(
            f"{marker} {row['Source IP']:>15} -> {str(row['Destination IP']):<15} "
            f"{str(row['Protocol']):<5} pkts={int(row['Packet Count']):<6} "
            f"dir={str(row['Traffic Direction']):<14} {verdict:<9} "
            f"threat={float(row['Threat Probability']):5.1f}% "
            f"{row['Attack Type'] if verdict == PREDICTION_MALICIOUS else ''}"
        )


def run_capture(interface: str, seconds: int, flow_timeout: float, csv_path: Optional[str],
                api_url: Optional[str], interval: float = 5.0) -> Dict[str, Any]:
    """Capture for ``seconds`` and report the scored flows."""
    sensor = FlowSensor(interface, flow_timeout=flow_timeout)
    sensor.start()
    print(f"[i] capturing on '{interface}' for {seconds}s - press Ctrl+C to stop")
    print(f"[i] {_capture_guidance() if not sensor.running() else 'capture running'}")

    collected: List[Dict[str, Any]] = []
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            time.sleep(min(interval, max(deadline - time.time(), 0.1)))
            records = sensor.collect()
            if not records:
                continue
            collected.extend(records)
            scored = score_records(records)
            _print_records(records, scored)
            if api_url:
                push_to_api(api_url, records)
    except KeyboardInterrupt:
        print("\n[i] interrupted")
    finally:
        sensor.stop()
        remaining = sensor.aggregator.drain()
        if remaining:
            collected.extend(remaining)
            scored = score_records(remaining)
            _print_records(remaining, scored)
            if api_url:
                push_to_api(api_url, remaining)

    aggregator = sensor.aggregator
    summary = {
        "interface": interface,
        "seconds": seconds,
        "packets_seen": aggregator.packets_seen,
        "packets_ignored": aggregator.packets_ignored,
        "flows": len(collected),
    }
    if collected and csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        frame = score_records(collected)
        frame = frame if frame is not None else pd.DataFrame(collected)
        frame.to_csv(csv_path, index=False)
        summary["csv"] = csv_path
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Live traffic sensor for the threat detector.")
    parser.add_argument("--iface", help="Interface to monitor (use --list-interfaces to see options).")
    parser.add_argument("--seconds", type=int, default=60, help="Capture duration in seconds.")
    parser.add_argument("--flow-timeout", type=float, default=DEFAULT_FLOW_TIMEOUT,
                        help="Seconds of silence before a flow is considered finished.")
    parser.add_argument("--csv", help="Write every captured flow (with verdicts) to this CSV file.")
    parser.add_argument("--api", help="POST flow records to a running dashboard, e.g. "
                                     "http://127.0.0.1:5000/api/sensor/sample")
    parser.add_argument("--list-interfaces", action="store_true", help="List capture interfaces and exit.")
    parser.add_argument("--check", action="store_true",
                        help="Report whether live capture can run here, and exit.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check:
        print(_capture_guidance())
        return 0 if all(check["ok"] for check in check_prerequisites()) else 1

    if args.list_interfaces:
        for item in list_interfaces():
            print(f"  {item['name']:<40} [{item['os']}]")
        print("\nLive capture prerequisites:")
        print(_capture_guidance())
        return 0

    if not args.iface:
        parser.error("--iface is required (or use --list-interfaces)")

    try:
        get_predictor().require_model()
    except ModelNotAvailableError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    try:
        summary = run_capture(args.iface, args.seconds, args.flow_timeout, args.csv, args.api)
    except CaptureUnavailable as exc:
        print(f"[error] live capture unavailable: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nstopped")
        return 0

    print(
        f"[done] {summary['flows']} flows from {summary['packets_seen']} packets "
        f"on {summary['interface']} in {summary['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())