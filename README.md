# Python Radiation Portal Monitor (RPM) Simulator

## 1. Overview

This project is a command-line-based Radiation Portal Monitor (RPM) simulator, converted from the original TypeScript project developed by Sandia National Laboratories. The program simulates the data streams from one or multiple RPMs, each operating in its own "lane."

The primary purpose is to provide a configurable and realistic source of RPM data for testing, development, and integration of monitoring systems without the need for physical hardware. The simulator runs continuously, generating a mix of background radiation data and random vehicle occupancies, which may include gamma, neutron, or combined alarms based on statistical probabilities.

## 2. Features

* **Multi-Lane Simulation**: Run dozens of independent RPM simulators simultaneously, each representing a different lane.
* **Configurable via JSON**: All simulation parameters are controlled through an external `settings.json` file, allowing for easy configuration without code changes.
* **Realistic Data Generation**: Procedurally generates radiation profiles for vehicle occupancies using a Gaussian model to simulate the rise and fall of detector counts.
* **TCP/IP Data Streaming**: Each simulated RPM lane listens on a dedicated TCP port, streaming data in a format compatible with real-world RPM systems.
* **Automatic Occupancy Mode**: Automatically generates random vehicle occupancies at configurable intervals.
* **Probabilistic Alarms**: The type of each occupancy (e.g., normal, gamma alarm, neutron alarm) is determined by user-defined probabilities.
* **Live Console Monitoring**: Provides a clean, real-time status dashboard in the console, showing the state of each lane.
* **File Logging**: Optionally logs all events and status changes to a file for later analysis.

## 3. Requirements

* Python 3.7+
* Required Python libraries:
    * `numpy`
    * `scipy`

## 4. Installation

1.  **Save the Script**: Place the `rpm_simulator.py` script in a directory.

2.  **Install Dependencies**: Open a terminal or command prompt and install the required libraries using `pip`.
    ```bash
    pip install numpy scipy
    ```

## 5. Configuration

The entire simulation is controlled by a `settings.json` file located in the same directory as the script.

### 5.1. `settings.json` Structure

Here is an example configuration for two lanes. You can add as many lane objects to the `"Lanes"` list as needed.

```json
{
    "Version": "1.0.0",
    "LogLevel": "INFO",
    "LogFilename": "rpm_simulator.log",
    "Lanes": [
        {
            "LaneID": 1,
            "LaneName": "Lane 1 (Local)",
            "Enabled": true,
            "AutoGammaProbability": 0.25,
            "AutoNeutronProbability": 0.10,
            "AutoInterval": 45,
            "RPM": {
                "IPAddr": "127.0.0.1",
                "Port": 10001,
                "GammaBG": 250,
                "NeutronBG": 2,
                "GammaNSigma": 6,
                "NeutronThreshold": 5,
                "GHThreshold": 450,
                "GLThreshold": 80,
                "NHThreshold": 10
            }
        },
        {
            "LaneID": 2,
            "LaneName": "Lane 2 (Remote)",
            "Enabled": true,
            "AutoGammaProbability": 0.05,
            "AutoNeutronProbability": 0.05,
            "AutoInterval": 60,
            "RPM": {
                "IPAddr": "0.0.0.0",
                "Port": 10002,
                "GammaBG": 220,
                "NeutronBG": 3,
                "GammaNSigma": 5,
                "NeutronThreshold": 6,
                "GHThreshold": 400,
                "GLThreshold": 100,
                "NHThreshold": 12
            }
        }
    ]
}
```

### 5.2. Parameter Definitions

#### Global Settings
* `"LogLevel"`: The level of detail for console and file logs. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`.
* `"LogFilename"`: The name of the file to write logs to. If set to `null` or an empty string, logs will only be written to the console.

#### Lane Settings (`Lanes` array)
Each object in the `"Lanes"` list defines one simulator instance.

* `"LaneName"`: A descriptive name for the lane (e.g., "East Gate - Inbound").
* `"Enabled"`: If `true`, this lane's simulator will start. If `false`, it will be ignored.
* `"AutoInterval"`: The average time in **seconds** between the end of one occupancy and the start of the next.
* `"AutoGammaProbability"`: The probability (from 0.0 to 1.0) that a generated occupancy will contain a gamma source.
* `"AutoNeutronProbability"`: The probability (from 0.0 to 1.0) that a generated occupancy will contain a neutron source.

#### RPM Settings (nested within each Lane)
These parameters control the physics and data output of the RPM.

* `"IPAddr"`: The IP address the RPM server will bind to. Use `"127.0.0.1"` for local access only, or `"0.0.0.0"` to allow connections from other computers on the network.
* `"Port"`: The TCP port for the data stream. **Must be unique for each lane.**
* `"GammaBG"`: The average 1-second background count rate for gamma detectors.
* `"NeutronBG"`: The average 1-second background count rate for neutron detectors.
* `"GammaNSigma"`: The number of standard deviations (N-Sigma) above background required to trigger a gamma alarm during an occupancy.
* `"NeutronThreshold"`: The absolute counts-per-second required to trigger a neutron alarm.
* `"GHThreshold"`: **G**amma **H**igh threshold. Triggers a `GH` background message if the count exceeds this value.
* `"GLThreshold"`: **G**amma **L**ow threshold. Triggers a `GL` background message if the count falls below this value.
* `"NHThreshold"`: **N**eutron **H**igh threshold. Triggers an `NH` background message if the count exceeds this value.

## 6. How to Run

1.  Ensure your `settings.json` file is configured and saved in the same directory as the script.
2.  Run the simulator from your terminal:
    ```bash
    python rpm_simulator.py
    ```
3.  The program will start and display a status dashboard, which updates periodically.
    ```
    --- Sandia Radiation Portal Monitor (Python) ---
    2025-06-09 23:30:00,123 - [MainThread  ] - INFO     - Logging configured with level INFO
    ...
    2025-06-09 23:30:00,456 - [Lane 1 (Local)] - INFO     - Starting lane...
    2025-06-09 23:30:00,789 - [Lane 2 (Remote)] - INFO     - Starting lane...

    ========================================
    Status at 2025-06-09 23:30:15
    ========================================
      Lane: Lane 1 (Local)  | Status: running      | Clients: 0   | Occupancy: unoccupied
      Lane: Lane 2 (Remote) | Status: running      | Clients: 0   | Occupancy: unoccupied
    ========================================
    (Press Ctrl+C to stop the simulator)
    ```
4.  To stop the simulator, press `Ctrl+C`. The program will perform a graceful shutdown of all running lanes.

## 7. Connecting to the Data Stream

You can connect to the raw TCP data stream of any running lane using a network utility like `netcat` (`nc`) or `telnet`.

For example, to connect to **Lane 1** as configured above, use the following command:

```bash
# On Linux or macOS
nc 127.0.0.1 10001

# On Windows (using telnet)
telnet 127.0.0.1 10001
```

You will see the raw data messages (e.g., `GB`, `NB`, `GS`, `GA`) being printed to your terminal as they are generated by the simulator.

## 8. Acknowledgments

This program is a Python implementation of the original **SRLS (Simulated Radiation Location Sensor)** project created by **Sandia National Laboratories**.

* **Original Project**: [https://github.com/sandialabs/SRLS](https://github.com/sandialabs/SRLS)