# Kyaro Anti-DDoS

A comprehensive, multi-layered DDoS protection system that integrates machine learning for advanced threat detection.

## Features

- **Multi-Layer Protection**: Defends against attacks at Layer 3 (Network), Layer 4 (Transport), and Layer 7 (Application)
- **Machine Learning Integration**: Advanced threat detection using machine learning models
- **Real-time Monitoring**: Dashboard for visualizing traffic patterns and attack metrics
- **API Access**: RESTful API for integration with other security tools
- **Configurable Rules**: Fine-tune protection settings through a user-friendly interface
- **Cross-Platform**: Works on both Windows and Linux

## Architecture

Kyaro Anti-DDoS consists of:

1. **Rust Backend**: High-performance core for packet processing and protection logic
2. **Python ML Engine**: Machine learning models for threat detection
3. **React Dashboard**: User interface for monitoring and configuration
4. **API Server**: REST API for integrations and programmatic access

## Protection Mechanisms

### Layer 3 Protection
- IP reputation checking
- Bogon filtering
- Geolocation-based filtering
- Traffic rate limiting
- Packet anomaly detection

### Layer 4 Protection
- SYN flood protection
- UDP flood protection
- TCP connection tracking
- Port scanning detection
- Protocol validation

### Layer 7 Protection
- HTTP request rate limiting
- Web Application Firewall (WAF)
- Bot detection
- Request pattern analysis
- DDoS fingerprinting

## Getting Started

### Prerequisites

- Rust (1.70 or newer)
- Python 3.9+ (with pip)
- Node.js 18+ (for dashboard)
- Administrative privileges (for packet capture)

### Installation

#### Windows

1. Clone the repository:
   ```
   git clone https://github.com/kyaro/antiddos.git
   cd kyaro-antiddos
   ```

2. Run the installation script:
   ```
   install.bat
   ```

#### Linux

1. Clone the repository:
   ```
   git clone https://github.com/kyaro/antiddos.git
   cd kyaro-antiddos
   ```

2. Run the installation script:
   ```
   chmod +x install.sh
   sudo ./install.sh
   ```

### Starting the Service

#### Windows

```
start.bat
```

#### Linux

```
sudo ./start.sh
```

### Stopping the Service

#### Windows

```
stop.bat
```

#### Linux

```
sudo ./stop.sh
```

## Configuration

Configuration is managed through:

1. The dashboard UI at `http://localhost:3000/configuration`
2. Editing the `config.json` file directly

### Sample Configuration

```json
{
  "api_port": 8080,
  "log_level": "info",
  "protection": {
    "layer3": {
      "enabled": true,
      "packet_rate_threshold": 1000,
      "blacklist_duration_seconds": 300
    },
    "layer4": {
      "enabled": true,
      "syn_flood_threshold": 100,
      "udp_flood_threshold": 200
    },
    "layer7": {
      "enabled": true,
      "request_rate_threshold": 50,
      "waf_enabled": true
    }
  },
  "ml": {
    "enabled": true,
    "model_path": "models/threat_detection.pkl",
    "threshold": 0.8
  }
}
```

## Dashboard

The dashboard provides:

- Real-time traffic visualization
- Attack detection alerts
- Configuration management
- Historical data analysis
- Blacklist management

Access the dashboard at `http://localhost:3000` after starting the service.

## API Documentation

The API server runs on port 8080 by default and provides endpoints for:

- `/api/status` - System status
- `/api/stats` - Traffic statistics
- `/api/blacklist` - IP blacklist management
- `/api/config` - Configuration management
- `/api/logs` - Log access

For detailed API documentation, visit `http://localhost:8080/api/docs` after starting the service.

## Development

### Building from Source

#### Backend

```
cd kyaro_antiddos
cargo build --release
```

#### Dashboard

```
cd dashboard
npm install
npm run build
```

### Running Tests

```
cargo test
python -m pytest ml/tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [tokio](https://tokio.rs/) - Asynchronous runtime for Rust
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for Python
- [React](https://reactjs.org/) - UI library for the dashboard 