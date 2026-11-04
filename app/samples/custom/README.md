# Flow files from our own traffic captures

These CSV files are flows from traffic the team captured on its own laptop, turned into the
final models' 23 features by `scripts/pcap_to_csv.py`. They are for the demo only: upload one
with **Upload a CSV** in `app/demo.py`. They are not used for training, tuning or any score in
the report.

Expect lower accuracy than in the report (0.85-0.94 test macro F1): the models learned from the
2016 ISCXVPN2016 captures, and today's apps and VPNs send traffic differently.

## Which task to choose for a file

Each file name ends with the task it is made for, and its `label` column holds that task's class
names. Upload it under exactly that task; under another task the columns may fit but the labels
do not (an `_A2-NonVPN` file uploaded under A2-VPN shows 0 correct).

| File ends with | Choose task | Labels in the file |
|---|---|---|
| `_A1.csv` | A1 | `VPN` or `Non-VPN` |
| `_B.csv` | B | e.g. `VPN-STREAMING` or `STREAMING` |
| `_A2-VPN.csv` | A2-VPN | e.g. `VPN-STREAMING` |
| `_A2-NonVPN.csv` | A2-NonVPN | e.g. `STREAMING` |

## How the converter was checked

`scripts/check_pcap_to_csv.py` converts the dataset's own captures (`datasets/PCAPs/`) and looks
for every flow in the dataset's ARFF files: `reports/pcap_to_csv_check.txt` (15 s timeout) and
`reports/pcap_to_csv_check_120s.txt` (120 s). Of the flows with 3 or more packets whose duration
occurs in the A2 15 s file, 94.9% are reproduced on all 23 values (90.9% at 120 s). Many flows of
a few captures are not in the files at all. In the differing flows checked by hand, the converter
counted a few backward packets that the original tool did not.

The converter follows the dataset's tool (ISCXFlowMeter): IPv4 TCP and UDP only (other packets,
including IPv6, are skipped); two-way flows, the first packet's sender is forward; packets in file
order; a 15 s flow timeout counted from a flow's first packet; a TCP FIN ends a flow; IP total
length as bytes, as captured (a packet the network card combined before capture can be over 1,500
bytes, as in some dataset flows); gaps of more than 1 s are idle; and the tool's own definition of
an active period. Captured traffic is never stored in the repository: only the 23 numbers per
flow are (no addresses, ports or content).

## How to make new files

1. **Capture** with Wireshark on your own laptop and your own network only. Close other apps.
   One activity per capture, about 5 minutes. Record each activity twice, VPN off and VPN on:

   | Activity | Class |
   |---|---|
   | YouTube, Netflix or Spotify | STREAMING |
   | A voice call (Skype, WhatsApp, Discord) | VOIP |
   | Text chat (WhatsApp, Telegram, Facebook Messenger) | CHAT |
   | Sending and receiving mail in a mail program (Apple Mail or Thunderbird; not webmail in a browser, which is web traffic) | MAIL |
   | A large file transfer (SFTP/scp, or a file sent in Skype) | FT |
   | A BitTorrent client with a legal file (only where the network allows it) | P2P |
   | Normal web browsing | BROWSING |

   Which interface to record: **VPN off: Wi-Fi (`en0`). VPN on: the VPN's own interface**
   (`utun...` on macOS: the one that appears when the VPN connects). The dataset's VPN captures
   hold the traffic inside the tunnel, not the encrypted tunnel itself, so recording Wi-Fi with
   the VPN on would give different flows.

2. **Save** each capture as `captures/raw/<activity>_<vpn|novpn>_<YYYY-MM-DD>.pcapng`.
   `captures/` is git-ignored: a capture holds addresses, the names of sites visited and
   sometimes readable content, so it never goes into the repository.

3. **Convert** each capture once per task (flow timeout 15 s, as the final models use):

   ```bash
   uv run python scripts/pcap_to_csv.py captures/raw/youtube_vpn_2026-11-05.pcapng --task B --label VPN-STREAMING --output app/samples/custom/youtube_vpn_B.csv
   ```

   Make `_A1` (label `VPN`/`Non-VPN`), `_B`, and `_A2-VPN` (VPN on) or `_A2-NonVPN` (VPN off).
   The script refuses a label that is not a class of the chosen task.

4. **Record** each capture in the table below and try the files in the demo.

## Captures in this folder

| Files | Activity and app | VPN (app) | Date | Length | Flows |
|---|---|---|---|---|---|
| (none yet) | | | | | |
