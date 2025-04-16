# Kyaro Anti-DDoS

A comprehensive, automated, and intelligent DDoS protection system designed to defend against a wide range of attack vectors.

## Features

### Multi-Layer Protection

**Layer 3 (Network Layer) Protection:**
- IP Blacklisting/Whitelisting
- Geographic IP Blocking
- Rate Limiting
- Abnormal Packet Detection (size, TTL, IP options)
- Protocol Filtering
- Sinkholing/Blackholing

**Layer 4 (Transport Layer) Protection:**
- Port Blocking
- TCP Protection (SYN Flood, abnormal flags)
- UDP Flood Protection
- ICMP Flood Protection
- Stateful Packet Inspection

**Layer 7 (Application Layer) Protection:**
- Web Application Firewall (WAF)
- OWASP Top 10 Protection
- HTTP/HTTPS Rate Limiting
- Request Analysis
- Bot Detection
- DNS Protection
- SSL/TLS Protection

### AI/ML Integration

- Machine Learning-based traffic analysis
- Anomaly detection for zero-day attacks
- Automatic pattern recognition
- Adaptive defense mechanisms
- Continuous learning from attack data

### Automated Response

- Real-time attack detection
- Automatic mitigation deployment
- Dynamic rule generation
- IP reputation tracking
- Self-healing capabilities

### Dashboard

- Real-time monitoring
- Attack visualization
- Configuration management
- Event logging
- Alerts and notifications
- Custom reporting

## Architecture

Kyaro Anti-DDoS uses a hybrid architecture:

1. **Core Defense Engine** (Rust): High-performance packet processing, rule enforcement, and traffic filtering
2. **Machine Learning Module** (Python): AI-based attack detection and pattern recognition
3. **Web Dashboard** (TypeScript/React): Configuration, monitoring, and visualization interface

## Getting Started

### Prerequisites

- Rust 1.70+ (with Cargo)
- Python 3.8+
- Node.js 18+
- Administrative privileges (for packet capture)

### Installation

1. Clone the repository:
```
git clone https://github.com/kyaro/antiddos.git
cd kyaro_antiddos
```

2. Build the Rust core:
```
cargo build --release
```

3. Install Python dependencies:
```
cd src/ml
pip install -r requirements.txt
```

4. Build the dashboard:
```
cd ../../dashboard
npm install
npm run build
```

5. Run the system:
```
cargo run --release
```

## Configuration

Configuration is stored in a JSON file and can be edited directly or through the web dashboard. The default location is `config.json`.

## Usage

The system runs as a service that protects your network infrastructure. Access the dashboard at `http://localhost:6868` to monitor and configure the system.

### Command-line Options

```
kyaro_antiddos [OPTIONS]

Options:
  -c, --config <FILE>   Path to config file [default: config.json]
  -p, --port <PORT>     Dashboard port [default: 6868]
  -l, --log <LEVEL>     Log level [default: info]
  -h, --help            Print help information
  -V, --version         Print version information
```

## Development

### Project Structure

- `src/backend/` - Rust core functionality
- `src/ml/` - Python machine learning components
- `src/common/` - Shared type definitions
- `dashboard/` - Web dashboard

### Building from Source

Follow the installation instructions above to build from source.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kyaro Team
- Open-source security community
