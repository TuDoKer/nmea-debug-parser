# NMEA Debug Parser

A PyQt5 desktop tool for parsing NMEA 0183 debug dumps and analyzing GNSS
receiver static observations. Pure offline, no network calls.

## Features

- Parse `.dbg` / `.log` / `.txt` dumps into typed NMEA messages (GGA, GSA,
  GST, RMC, GSV, and vendor-extended messages) plus a separate binary stream.
- Export selected message types to CSV with latitude / longitude / altitude
  split into columns.
- GGA statistics: mean, median, CEP95, RMSE, and time-series plots.
- RINEX 2/3 observation and broadcast ephemeris parsing.
- Static observation quality analysis (MP combination, ionospheric ROT/IROT,
  elevation correlation, per-constellation satellites).
- Single-point positioning (SPP) elevation time series.
- Optional two-station static baseline solving, driven by an external RTKLIB
  `rnx2rtkp.exe`.

## Install & run

```bash
pip install -r requirements.txt
python nmea_debug_parser.py
```

Requires Python 3.8+ (developed on 3.13), PyQt5 >= 5.15, numpy >= 1.20,
matplotlib >= 3.5. Tested on Windows; should run on Linux/macOS with the
same dependencies.

## Optional: static baseline solving

The relative-positioning tab calls an external `rnx2rtkp.exe` from RTKLIB
(2.4.3 b34 or demo5). It is **not bundled**. Provide it via the `--rtklib`
option or the in-app path field. All other features work without it.

## Project layout

```
nmea_debug_parser.py     entry point
nmea_parser/
  parser.py              NMEA message parsing (pure logic)
  statistics.py          GGA / GSV statistical analysis
  geodesy.py             coordinate transforms (single implementation)
  rinex*.py              RINEX observation / ephemeris / quality analysis
  solfile.py             .pos / .llh solution file parsing
  spp.py                 single-point positioning
  baseline.py            external RTKLIB driver
  ui/                    PyQt5 views and dialogs
```

## License

MIT. See [LICENSE](LICENSE).
