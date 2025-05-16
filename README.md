# BLE Post Processing for position estimation

This repository implements position estimation using BLE packets by filtering raw RSSI values, estimating distances based on different algorithms, and lastly estimating the position based on the estimated distances.

All programs are written in Python, they require Python3.13, PySerial, matplotlib and QT6 to be available.

All programs have CLI support, so for a list of possible arguments just add `-h`

## Structure

- `nrf_sniffer/`
  + GUI Application for receiving, filtering and displaying received BLE packets by listening to an nRF Sniffer API enabled dongle like the nRF52840.
  + It contains a custom built backend running on a separate python process, this way the capturing wont be negatively impacted by the GUI processing.
- `processor/`
  + Full processing of raw data, handling static and dynamic sessions, by applying all the processors (Triples for dynamic and Pairs for static), it performs parameter calibration and produces plots used in the thesis.
- `shared/`
  + Contains all the shared components of the project, this includes definitions for MACAddress and a higher accuracy timer than pythons standard variant.
  + It also contains definitions for the source data and processed data, it has the data parser which handles loading and storing sessions to disk.
  + It also contains all the processor implementations inside `shared/Processors/`. Which are all implemented with numpy vectorization to not slow down the processing much.
- `tests/`
  + Contains all the captured tests, full structure info in [README.md](tests/README.md)

## Entire process
First start the `nrf_sniffer` application specifying the name for your test, connect the nRF52840 dongle to your pc, ensure beacons are nearby and their MACAddresses are defined in the code (Line 253 in `nrf_sniffer/Figures/RSSIFigure.py`). Now start capturing for however long you want to capture for, once the capture is done the output will be stored in `tests/{testname}/`. Once you have collected all the data you need run the `processor` program specifying the name for your test, it will run for either a few minutes or possibly a few hours, depending on the size of your data. If you decide to modify an algorithm but want to retain the calibrated parameters add the `--skip-calibration` argument, if you want to skip processing as well because you only modified the plotting `--skip-processing`, if you only want to calibrate a single triple, skip all the other processors with `--skip-filter filterA filterB ...`, `--skip-distance distA distB ...`, `--skip-position posA posB ...`. For a full list of parameters just add `-h` to the command.