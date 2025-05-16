# Structure
*The following descriptions have hover text*  
[{testName}](# "Stores all data for this session.")
- [parameters.json](# "Stores session parameters.")
  ```
  {
    "Beacons": {
      "MACAddress": {
        "Name": "Name",
        "Height": 0.0,
        "PosX": 0.0,
        "PosY": 0.0,
        "NormX": 0.0,
        "NormY": 0.0
      }
    },
    "Sniffer": {
      "Height": 0.0,
      "dX": 0.0,
      "dY": 0.0
    },
    "Tunnel": {
      "Width": 10.0,
      "Depth": 100.0,
      "Shape": [
        [ 0.0, 0.0 ],
        [ 0.0, 100.0 ]
      ]
    },
    "WindowSizes": [ 60.0 ]
  }
  ```
- [static/](# "Stores the static data from the Sniffer for individual periods.")
  + [{PosX}_{PosY}/](# "Stores the samples captured at PosX, PosY.")
    - [{DeviceAddress}.csv](# "Stores the samples captured from DeviceAddress.")
      ```
      Timestamp,CPUTimestamp,RSSI,CRCOK
      ```
- [static_raw/](# "Stores the raw static data from the Sniffer for individual periods.")
  + [{PosX}_{PosY}/](# "Stores the samples captured at PosX, PosY.")
    - [{DeviceAddress}.csv](# "Stores the samples captured from DeviceAddress.")
      ```
      Timestamp,CPUTimestamp,RSSI,CRCOK,PDUType,Channel,AuxType,PHY,PacketCounter,AA,CI,PDU
      ```
- [dynamic/](# "Stores the dynamic data from the Sniffer for individual periods.")
  + [{Index}/](# "Stores the Index-nth period.")
    - [{DeviceAddress}.csv](# "Stores the samples captured from DeviceAddress.")
      ```
      Timestamp,CPUTimestamp,RSSI,CRCOK
      ```
- [dynamic_raw/](# "Stores the raw dynamic data from the Sniffer for individual periods.")
  + [{Index}/](# "Stores the Index-nth period.")
    - [{DeviceAddress}.csv](# "Stores the samples captured from DeviceAddress.")
      ```
      Timestamp,CPUTimestamp,RSSI,CRCOK,PDUType,Channel,AuxType,PHY,PacketCounter,AA,CI,PDU
      ```
- [static_cancelled/](# "Stores cancelled static periods.")
- [static_raw_cancelled/](# "Stores cancelled raw static periods.")
- [processed/](# "Stores the processed results")
  + [static_full/](# "Contains optimized parameters and raw processed outputs for each pair of processor")
    - [{Distance}_parameters.json](# "Parameters for the Distance estimator")
    - [{Distance}_{Position}/](# "Processed data associated with Distance and Position processors")
      + [distances/](# "Contains the generated distributions per position")
      + [positions/](# "Contains the generated distributions per position")
  + [static_good/](# "Same as static_full, but only for good packets")
  + [static_full_validation](# "Contains validational outputs, distMSE and posMSE files")
  + [static_good_validation](# "Same as static_full_validation")
  + [static_full_plots](# "Contains all the produced plots, similar structure as static_full")
  + [static_good_plots](# "Same as static_full_plots")
  + [dynamic_full/](# "Contains optimized parameters and raw processed outputs for each triple of processors")
    - [{Filter}_{Distance}_parameters.json](# "Parameters for the combination of Filter and Distance estimator")
    - [{Filter}\_{Distance}\_{Position}/](# "Processed data associated with the Filter, Distance and Position combination")
      + [filtered/](# "Contains the graphs of filtered output")
      + [distances/](# "Contains the graphs of distance outputs")
      + [positions/](# "Contains the graphs of position outputs")
  + [dynamic_good/](# "Same as dynamic_full, but only for good packets")
  + [dyanmic_full_validation](# "Contains validational outputs, distMSE and posMSE files")
  + [dynamic_good_validation](# "Same as dynamic_full_validation")
  + [dynamic_full_plots](# "Contains all the produced plots, similar structure as dynamic_full")
  + [dynamic_good_plots](# "Same as dynamic_full_plots")