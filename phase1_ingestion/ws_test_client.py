"""
ws_test_client.py — WebSocket Test Client

Simple client to verify the WebSocket alert stream before wiring
the React frontend. Connects to ws://localhost:8000/ws/alerts and
prints incoming alert messages.

Usage:
    python -m phase1_ingestion.ws_test_client
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("ERROR: websockets is required. Install with: pip install websockets")
    sys.exit(1)


async def listen(url: str = "ws://localhost:8001/ws/alerts"):
    """Connect to the alert WebSocket and print incoming messages."""
    print(f"🔌 Connecting to {url}...")

    try:
        async with websockets.connect(url) as ws:
            print("✅ Connected! Waiting for alerts...\n")

            async for message in ws:
                try:
                    alert = json.loads(message)
                    slice_id = alert.get("slice_id", "?")
                    flagged = alert.get("flagged", False)
                    findings = alert.get("findings", [])
                    proc_time = alert.get("processing_time_ms", 0)
                    has_thumb = alert.get("thumbnail") is not None

                    if flagged:
                        print(f"  🔴 Slice {slice_id}: FLAGGED ({len(findings)} findings, {proc_time:.1f}ms)")
                        for f in findings:
                            print(f"     → {f['finding_type']}: bbox={f['bbox']}, HU={f['hu_mean']:.0f}")
                    else:
                        print(f"  🟢 Slice {slice_id}: clean ({proc_time:.1f}ms)")

                    if has_thumb:
                        thumb_len = len(alert["thumbnail"])
                        print(f"     📸 Thumbnail: {thumb_len} chars (base64)")

                except json.JSONDecodeError:
                    print(f"  ⚠️  Non-JSON message: {message[:100]}")

    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the pipeline is running:")
        print("   python -m phase1_ingestion.pipeline")
    except KeyboardInterrupt:
        print("\n👋 Disconnected.")


def main():
    print("=" * 50)
    print("  PRISM WebSocket Test Client")
    print("=" * 50)
    asyncio.run(listen())


if __name__ == "__main__":
    main()
