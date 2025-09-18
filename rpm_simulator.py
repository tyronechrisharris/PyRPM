"""
SRLS - Radiation Portal Monitor Simulator in Python

This script simulates one or more Radiation Portal Monitors (RPMs) based on a 
JSON configuration file. It sets up TCP servers for each RPM to stream simulated 
detector data.

This program is a Python conversion of the original TypeScript project 
from https://github.com/sandialabs/SRLS.

Created on: 2024-06-07
"""
import json
import logging
import math
import socketserver
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import norm

#region 1. Logging Setup

class ELogLevel(Enum):
    """Enumeration for log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

def setup_logging(level: str = 'INFO', filename: Optional[str] = None):
    """Configures the root logger for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Basic configuration
    log_format = '%(asctime)s - [%(threadName)-12s] - %(levelname)-8s - %(message)s'
    
    # Use basicConfig to set up handlers; force=True allows re-configuration
    logging.basicConfig(level=log_level, format=log_format, force=True)

    # If a filename is provided, add a FileHandler
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    logging.info("Logging configured with level %s", level)
    if filename:
        logging.info("Log output is also directed to %s", filename)

#endregion

#region 2. Settings and Configuration

@dataclass
class RPMSettings:
    """Settings for a single RPM unit."""
    IPAddr: str
    Port: int
    GammaBG: float
    NeutronBG: float
    GammaNSigma: float
    NeutronThreshold: float
    GHThreshold: float
    GLThreshold: float
    NHThreshold: float
    GammaDistribution: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    NeutronDistribution: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

@dataclass
class LaneSettings:
    """Settings for a single simulated lane."""
    LaneID: int
    LaneName: str
    Enabled: bool
    AutoGammaProbability: float
    AutoNeutronProbability: float
    AutoInterval: int  # in seconds
    RPM: RPMSettings
    
    # Non-persistent state attributes
    ClientCount: int = 0
    Status: str = "stopped"
    OccupancyState: str = "unoccupied"

    # Allow post-init processing to handle nested dataclass from dict
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LaneSettings':
        rpm_settings = RPMSettings(**data['RPM'])
        return cls(
            LaneID=data['LaneID'],
            LaneName=data['LaneName'],
            Enabled=data['Enabled'],
            AutoGammaProbability=data['AutoGammaProbability'],
            AutoNeutronProbability=data['AutoNeutronProbability'],
            AutoInterval=data['AutoInterval'],
            RPM=rpm_settings
        )

@dataclass
class AppSettings:
    """Top-level application settings."""
    Version: str
    LogLevel: str
    LogFilename: Optional[str]
    Lanes: List[LaneSettings]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        lanes = [LaneSettings.from_dict(lane_data) for lane_data in data.get('Lanes', [])]
        return cls(
            Version=data['Version'],
            LogLevel=data['LogLevel'],
            LogFilename=data.get('LogFilename'),
            Lanes=lanes
        )

class SettingsManager:
    """Manages loading and saving application settings."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: AppSettings = self.load()

    def load(self) -> AppSettings:
        """Loads settings from the JSON file or returns defaults."""
        try:
            with open(self.filepath, 'r') as f:
                settings_dict = json.load(f)
                logging.info("Loaded settings from %s", self.filepath)
                return AppSettings.from_dict(settings_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning("Could not load %s (%s). Using default settings.", self.filepath, e)
            return self.default_settings()

    def save(self):
        """Saves current settings to the JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                # Convert dataclasses to dicts for JSON serialization
                json.dump(asdict(self.data), f, indent=4)
            logging.info("Saved settings to %s", self.filepath)
        except IOError as e:
            logging.error("Failed to save settings to %s: %s", self.filepath, e)

    def default_settings(self) -> AppSettings:
        """Provides a default configuration."""
        default_rpm = RPMSettings(
            IPAddr="127.0.0.1", Port=10001, GammaBG=250, NeutronBG=2,
            GammaNSigma=6, NeutronThreshold=5, GHThreshold=450,
            GLThreshold=80, NHThreshold=10
        )
        default_lane = LaneSettings(
            LaneID=1, LaneName="Default Lane", Enabled=True,
            AutoGammaProbability=0.1, AutoNeutronProbability=0.05,
            AutoInterval=30, RPM=default_rpm
        )
        return AppSettings(
            Version="1.0.0", LogLevel="INFO", LogFilename="rpm_simulator.log",
            Lanes=[default_lane]
        )

#endregion

#region 3. Core Components

class Component:
    """Base class for simulation components, providing logging."""
    def __init__(self, name: str, level: ELogLevel = ELogLevel.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
    
    def log(self, msg: str, level: ELogLevel = ELogLevel.INFO):
        self._logger.log(level.value, msg)

@dataclass
class DetectorValues:
    """Represents detector counts at a specific time."""
    msg_type: str
    time_offset: float  # in milliseconds
    values: List[int]
    msg_time: float = 0.0  # absolute epoch time for sending

    @property
    def total_counts(self) -> int:
        return sum(self.values)

    def __str__(self):
        """Formats the data into an RPM-compatible string."""
        if self.msg_type == "SP":
            return "SP,0.2249,03.032,004.88,000000"
        
        counts_str = ",".join([f"{v:06d}" for v in self.values])
        return f"{self.msg_type},{counts_str}"

class RPMProfile:
    """Manages a sequence of detector values for an occupancy event."""
    def __init__(self):
        self.counts: List[DetectorValues] = []
        self.type: str = "oc"  # occupancy, ga, na, ng
        self._cursor: int = 0
    
    def add_sample(self, vals: DetectorValues):
        """Adds a new sample to the profile and determines alarm type."""
        self.counts.append(vals)
        if vals.msg_type == "GA":
            self.type = "ng" if self.type in ["na", "ng"] else "ga"
        elif vals.msg_type == "NA":
            self.type = "ng" if self.type in ["ga", "ng"] else "na"

    def add_gx(self, counter: int):
        """Adds a GX (end of occupancy) message."""
        if self.counts:
            last_msg_time = self.counts[-1].msg_time
            gx_vals = DetectorValues("GX", last_msg_time, [counter, counter * 10, 0, 0])
            self.add_sample(gx_vals)

    def add_time_offset(self, offset: float):
        """Sets absolute send times for all messages in the profile."""
        for p in self.counts:
            p.msg_time = offset + (p.time_offset / 1000.0)
        self._cursor = 0

    def is_eof(self) -> bool:
        """Checks if all messages in the profile have been processed."""
        return self._cursor >= len(self.counts)

    def get_next_message(self, now: float) -> Optional[DetectorValues]:
        """Returns the next message if its scheduled time has passed."""
        if self.is_eof():
            return None
        
        dv = self.counts[self._cursor]
        if now >= dv.msg_time:
            self._cursor += 1
            return dv
        return None

class ProfileGenerator(Component):
    """Generates synthetic radiation profiles."""
    def __init__(self):
        super().__init__("ProfileGenerator")
        
    def generate_profile(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Generates a Gaussian-shaped profile based on input parameters.
        
        Args:
            params: A dictionary with keys like 'duration', 'stddev', 'humps', 'shift'.
        """
        duration = float(params.get('duration', 10.0))
        stddev = float(params.get('stddev', duration / 2.0))
        time_increment = int(params.get('time_increment', 1))
        humps = int(params.get('humps', 1))
        shift = float(params.get('shift', 0.0))

        # Generate time values (x-axis)
        # Duration * 5 to ensure the bell curve starts and ends near zero
        x_values = np.arange(0, duration * 5, time_increment)
        
        # Calculate mean for the Gaussian distribution
        mean = (x_values[0] + x_values[-1]) / 2.0
        
        y_values = np.zeros_like(x_values, dtype=float)

        if humps > 1:
            # Two humps, shifted symmetrically from the center
            mean1 = mean * (1.0 - shift)
            mean2 = mean * (1.0 + shift)
            y_values += norm.pdf(x_values, loc=mean1, scale=stddev)
            y_values += norm.pdf(x_values, loc=mean2, scale=stddev)
        else:
            # Single hump, shifted from the center
            shifted_mean = mean * (1.0 + shift)
            y_values = norm.pdf(x_values, loc=shifted_mean, scale=stddev)

        # Normalize y_values to a 0-1 range
        if np.max(y_values) > 0:
            y_values /= np.max(y_values)

        return np.column_stack((x_values, y_values))

#endregion

#region 4. RPM Simulator

class TCPHandler(socketserver.BaseRequestHandler):
    """Handler for incoming TCP connections to an RPM."""
    def handle(self):
        self.server.add_client(self)
        self.server.log(f"Client connected: {self.client_address}")
        try:
            # Keep the connection open but don't read data
            while True:
                time.sleep(1)
        except Exception:
            # This will catch BrokenPipeError, etc. when the client disconnects
            pass
        finally:
            self.server.remove_client(self)
            self.server.log(f"Client disconnected: {self.client_address}")

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer, Component):
    """A threaded TCP server that can manage clients."""
    daemon_threads = True
    allow_reuse_address = True
    
    def __init__(self, server_address, RequestHandlerClass, rpm_simulator):
        Component.__init__(self, f"TCPServer-{server_address[1]}")
        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass)
        self.rpm_simulator = rpm_simulator
        self.clients = []
        self.clients_lock = threading.Lock()

    def add_client(self, client):
        with self.clients_lock:
            self.clients.append(client)
    
    def remove_client(self, client):
        with self.clients_lock:
            if client in self.clients:
                self.clients.remove(client)
    
    def broadcast(self, message: str):
        """Sends a message to all connected clients."""
        with self.clients_lock:
            doomed = []
            for client in self.clients:
                try:
                    client.request.sendall(message.encode('utf-8'))
                except (BrokenPipeError, ConnectionResetError):
                    doomed.append(client)
            for client in doomed:
                self.remove_client(client)

class RPMSimulator(Component):
    """Simulates a single Radiation Portal Monitor."""
    def __init__(self, name: str, settings: LaneSettings, owner: 'LaneSimulator'):
        super().__init__(name)
        self._owner = owner
        self._settings = settings.RPM
        self._auto_settings = {
            'prob_gamma': settings.AutoGammaProbability,
            'prob_neutron': settings.AutoNeutronProbability,
            'interval_s': settings.AutoInterval
        }
        
        self._profile_generator = ProfileGenerator()
        self._queued_profiles: List[RPMProfile] = []
        self._current_profile: Optional[RPMProfile] = None
        
        self._gx_counter = 0
        self._next_background_time: float = time.time()
        self._background_interval_s = 5.0
        
        self._auto_mode_active = False
        self._auto_mode_next_occupancy_time: float = time.time()

        self._server: Optional[ThreadedTCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def start(self):
        """Starts the RPM simulator and its TCP server."""
        if self._running:
            return
        self.log(f"Starting RPM listener on {self._settings.IPAddr}:{self._settings.Port}")
        self._running = True
        
        server_address = (self._settings.IPAddr, self._settings.Port)
        self._server = ThreadedTCPServer(server_address, TCPHandler, self)
        
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, 
            name=f"RPM-{self._settings.Port}"
        )
        self._server_thread.daemon = True
        self._server_thread.start()
        
        self._timer = threading.Timer(0.01, self._on_timer)
        self._timer.start()
        self.log("RPM simulator started.")

    def stop(self):
        """Stops the RPM simulator."""
        if not self._running:
            return
        self.log("Stopping RPM simulator.")
        self._running = False

        if self._timer:
            self._timer.cancel()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread:
            self._server_thread.join()
        
        self.log("RPM simulator stopped.")

    def start_auto_mode(self):
        if not self._auto_mode_active:
            self._auto_mode_next_occupancy_time = time.time()
            self._auto_mode_active = True
            self.log("Auto mode started.")
    
    def stop_auto_mode(self):
        if self._auto_mode_active:
            self._auto_mode_active = False
            self.log("Auto mode stopped.")

    @property
    def occupancy_state(self) -> str:
        if self._current_profile:
            return self._current_profile.type
        return "unoccupied"
        
    @property
    def client_count(self) -> int:
        return len(self._server.clients) if self._server else 0

    def generate_from_model(self, model: Dict[str, Any]) -> RPMProfile:
        """Generates an RPMProfile based on a descriptive model."""
        self.log(f"Generating profile from model: {model}")
        
        # Create normalized profiles (0-1 range) for gamma and neutron
        gamma_curve = self._profile_generator.generate_profile(model)
        model['time_increment'] = 5 # Sparser neutron readings
        neutron_curve = self._profile_generator.generate_profile(model)
        
        # Calculate target counts for alarms
        gamma_bg_s = self._settings.GammaBG
        sqrt_gamma_bg = math.sqrt(gamma_bg_s)
        
        # n_sigma = (5*c - bg) / sqrt(bg)  =>  c = (n_sigma * sqrt(bg) + bg) / 5
        # 'c' is the 200ms count, 'bg' is the 1s background count
        max_gamma_count_200ms = (model['gamma_nsigma'] * sqrt_gamma_bg + gamma_bg_s) / 5.0
        gamma_offset = max_gamma_count_200ms - (gamma_bg_s / 5.0)
        
        gamma_alarm_threshold_1s = self._settings.GHThreshold
        neutron_alarm_threshold_1s = self._settings.NeutronThreshold
        
        # Merge gamma and neutron curves into a single timeline
        profile = RPMProfile()
        
        # Convert curves to lists of (time, value)
        gamma_points = list(map(tuple, gamma_curve))
        neutron_points = list(map(tuple, neutron_curve))
        
        while gamma_points or neutron_points:
            # Determine which event is next in time
            take_gamma = False
            if gamma_points and neutron_points:
                if gamma_points[0][0] <= neutron_points[0][0]:
                    take_gamma = True
            elif gamma_points:
                take_gamma = True

            if take_gamma:
                time_idx, amplitude = gamma_points.pop(0)
                
                # Calculate the total gamma count for this 200ms slice
                # Base count is 1/5th of 1s background
                # Amplitude (0-1) scales the calculated offset
                total_count_200ms = (gamma_bg_s / 5.0) + (amplitude * gamma_offset)
                
                # Distribute total count across 4 detectors
                counts = self._distribute_counts(
                    total_count_200ms, 
                    self._settings.GammaDistribution
                )
                
                is_alarm = total_count_200ms * 5 > gamma_alarm_threshold_1s
                msg_type = "GA" if is_alarm else "GS"
                
                dv = DetectorValues(msg_type, time_idx * 200, counts)
                profile.add_sample(dv)

                # Add SP message every 5th reading (once per second)
                if len(profile.counts) % 5 == 0:
                    profile.add_sample(DetectorValues("SP", time_idx * 200, [0,0,0,0]))
            
            else: # Take neutron
                time_idx, amplitude = neutron_points.pop(0)
                
                # Neutron amplitude is scaled above background
                total_count_1s = self._settings.NeutronBG + (amplitude * model['neutron_amplitude'])
                counts = self._distribute_counts(
                    total_count_1s, 
                    self._settings.NeutronDistribution
                )

                is_alarm = total_count_1s > neutron_alarm_threshold_1s
                msg_type = "NA" if is_alarm else "NS"
                dv = DetectorValues(msg_type, time_idx * 200, counts)
                profile.add_sample(dv)
        
        # Add GX to signify end of occupancy
        self._gx_counter += 1
        profile.add_gx(self._gx_counter)
        
        self._queued_profiles.append(profile)
        return profile
    
    def _distribute_counts(self, total: float, distribution: List[float]) -> List[int]:
        """Distributes a total count across detectors with randomization."""
        counts = [round(total * w) for w in distribution]
        # Simple randomization: +/- 1 to preserve total
        if len(counts) > 1 and np.random.rand() > 0.5:
            idx1, idx2 = np.random.choice(len(counts), 2, replace=False)
            if counts[idx1] > 0:
                counts[idx1] -= 1
                counts[idx2] += 1
        return [max(0, c) for c in counts]
        
    def _on_timer(self):
        """Main periodic check, runs every ~10ms."""
        if not self._running:
            return
            
        try:
            self._tick()
        except Exception as e:
            self.log(f"Error in timer tick: {e}", ELogLevel.ERROR)
        finally:
            if self._running:
                # Reschedule the timer
                self._timer = threading.Timer(0.01, self._on_timer)
                self._timer.start()

    def _tick(self):
        """The core logic driven by the timer."""
        now = time.time()

        # Check for a new profile if none is active
        if self._current_profile is None and self._queued_profiles:
            self._current_profile = self._queued_profiles.pop(0)
            self._current_profile.add_time_offset(now)
            self.log(f"Starting new profile of type: {self._current_profile.type}")

        if self._current_profile:
            # We are in an occupancy, send profile messages
            msg = self._current_profile.get_next_message(now)
            msgs_to_send = ""
            while msg:
                msgs_to_send += str(msg) + "\r\n"
                msg = self._current_profile.get_next_message(now)
            
            if msgs_to_send:
                self._broadcast(msgs_to_send)
                
            if self._current_profile.is_eof():
                self.log(f"Profile finished.")
                self._current_profile = None
                self._next_background_time = now + self._background_interval_s
                if self._auto_mode_active:
                    self._auto_mode_next_occupancy_time = now + self._auto_settings['interval_s']
        else:
            # Not in an occupancy, check for auto-mode trigger or send background
            if self._auto_mode_active and now >= self._auto_mode_next_occupancy_time:
                self._trigger_auto_occupancy()
                # Prevent re-triggering immediately
                self._auto_mode_next_occupancy_time = now + 99999
            elif now >= self._next_background_time:
                self._send_background_counts()
                self._next_background_time = now + self._background_interval_s

    def _trigger_auto_occupancy(self):
        """Generates an alarm based on auto-mode probabilities."""
        self.log("Auto-mode triggering new occupancy.")
        is_gamma = np.random.rand() <= self._auto_settings['prob_gamma']
        is_neutron = np.random.rand() <= self._auto_settings['prob_neutron']
        
        alarm_type = "OC"
        if is_gamma and is_neutron: alarm_type = "NG"
        elif is_gamma: alarm_type = "GA"
        elif is_neutron: alarm_type = "NA"
            
        self._owner.generate_alarm(alarm_type)

    def _send_background_counts(self):
        """Sends GB and NB background messages."""
        if self.client_count == 0:
            return
            
        # Neutron Background (NB)
        n_counts = self._distribute_counts(self._settings.NeutronBG, self._settings.NeutronDistribution)
        n_msg_type = "NH" if sum(n_counts) > self._settings.NHThreshold else "NB"
        n_dv = DetectorValues(n_msg_type, 0, n_counts)
        
        # Gamma Background (GB)
        g_counts = self._distribute_counts(self._settings.GammaBG, self._settings.GammaDistribution)
        g_total = sum(g_counts)
        g_msg_type = "GB"
        if g_total > self._settings.GHThreshold: g_msg_type = "GH"
        elif g_total < self._settings.GLThreshold: g_msg_type = "GL"
        g_dv = DetectorValues(g_msg_type, 0, g_counts)
        
        self._broadcast(f"{n_dv}\r\n{g_dv}\r\n")

    def _broadcast(self, message: str):
        if self._server:
            self._server.broadcast(message)

#endregion

#region 5. Lane Simulator

class LaneSimulator(Component):
    """Manages a single lane, containing an RPM."""
    def __init__(self, settings: LaneSettings):
        super().__init__(settings.LaneName)
        self.settings = settings
        self.name = settings.LaneName
        self.is_enabled = settings.Enabled
        self.rpm = RPMSimulator(f"{self.name}-RPM", settings, self)
        self.is_in_auto_mode = False
        
        self.log(f"Lane '{self.name}' initialized.")

    def start(self):
        if self.is_enabled:
            self.log("Starting lane...")
            self.rpm.start()
            self.settings.Status = "running"
        else:
            self.log("Lane is disabled, not starting.")
            self.settings.Status = "disabled"

    def stop(self):
        self.log("Stopping lane...")
        self.rpm.stop()
        self.settings.Status = "stopped"

    def poll_status(self):
        """Updates the status attributes of the lane."""
        self.settings.OccupancyState = self.rpm.occupancy_state
        self.settings.ClientCount = self.rpm.client_count

    def set_auto_mode(self, auto_on: bool):
        if self.is_enabled:
            self.is_in_auto_mode = auto_on
            if auto_on:
                self.log("Starting auto mode.")
                self.rpm.start_auto_mode()
            else:
                self.log("Stopping auto mode.")
                self.rpm.stop_auto_mode()

    def generate_alarm(self, alarm_type: str, duration_s: float = -1.0):
        """Triggers the generation of an alarm profile in the RPM."""
        if not self.is_enabled:
            return
        
        self.log(f"Generating '{alarm_type}' alarm.")
        
        if duration_s <= 0:
            duration_s = 7.0 + np.random.rand() * 10 # 7-17 seconds
        
        # Determine alarm parameters
        gamma_nsigma = 0.0
        if alarm_type in ["GA", "NG"]:
            gamma_nsigma = self.rpm._settings.GammaNSigma + np.random.rand() * 2
        
        neutron_amplitude = 0.0
        if alarm_type in ["NA", "NG"]:
            neutron_amplitude = self.rpm._settings.NeutronThreshold + np.random.rand() * 3
            
        model = {
            'type': alarm_type,
            'duration': duration_s,
            'stddev': np.random.rand() * duration_s * 0.5 + 2.0,
            'humps': 1 if np.random.rand() < 0.7 else 2,
            'shift': (0.5 - np.random.rand()) * 0.8,
            'gamma_nsigma': gamma_nsigma,
            'neutron_amplitude': neutron_amplitude,
        }
        
        self.rpm.generate_from_model(model)

#endregion

#region 6. Main Application

def main():
    """Main function to run the SRLS simulator."""
    print("--- Sandia Radiation Portal Monitor Simulator (Python) ---")
    
    # Load settings
    settings_mgr = SettingsManager("settings.json")
    app_settings = settings_mgr.data

    # Configure logging
    setup_logging(app_settings.LogLevel, app_settings.LogFilename)

    # Initialize lanes
    lanes = [LaneSimulator(s) for s in app_settings.Lanes if s.Enabled]

    if not lanes:
        logging.error("No enabled lanes found in settings.json. Exiting.")
        sys.exit(1)
    
    # Start simulators
    for lane in lanes:
        lane.start()
        # Start in auto mode by default
        lane.set_auto_mode(True)
        
    try:
        while True:
            print("\n" + "="*40)
            print(f"Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*40)
            for lane in lanes:
                lane.poll_status()
                print(
                    f"  Lane: {lane.name:<15} | "
                    f"Status: {lane.settings.Status:<10} | "
                    f"Clients: {lane.settings.ClientCount:<3} | "
                    f"Occupancy: {lane.settings.OccupancyState:<12}"
                )
            print("="*40)
            print("(Press Ctrl+C to stop the simulator)")
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\nShutdown signal received. Stopping simulators...")
    finally:
        for lane in lanes:
            lane.stop()
        logging.info("All simulators stopped. Goodbye.")
        print("Simulation finished.")

if __name__ == "__main__":
    main()

#endregion
