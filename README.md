# Satellite Change Detection System

A Flask-based web application for detecting changes in satellite images with automated alerts for illegal activities.

## Features

- **Change Detection**: Compare before/after satellite images- **Visual Change Display**: See highlighted changes with red overlays- **Land Cover Classification**: Automatic classification of land types
- **Illegal Activity Alerts**:
  - Illegal Construction (Forest → Building)
  - Lake Encroachment (Water → Land)
  - Deforestation (Forest → Bare Land)
- **Multi-Channel Notifications**:
  - Web interface alerts with sound
  - Browser notifications
  - Email notifications
  - SMS notifications (Twilio)

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Configuration

### Email Setup (Gmail)
1. Enable 2-Factor Authentication on your Gmail account
2. Generate an App Password: https://myaccount.google.com/apppasswords
3. Update `.env`:
   ```
   ALERT_EMAIL=your-email@gmail.com
   ALERT_EMAIL_PASSWORD=your-app-password
   ALERT_RECEIVER_EMAIL=admin@example.com
   ```

### SMS Setup (Twilio)
1. Sign up for Twilio: https://www.twilio.com/
2. Get your Account SID, Auth Token, and phone numbers
3. Update `.env`:
   ```
   TWILIO_ACCOUNT_SID=your-sid
   TWILIO_AUTH_TOKEN=your-token
   TWILIO_FROM_NUMBER=+1234567890
   TWILIO_TO_NUMBER=+0987654321
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open http://127.0.0.1:5000 in your browser

3. Upload "before" and "after" satellite images

4. The system will:
   - Detect changes
   - Classify land cover types
   - Generate alerts for illegal activities
   - Send notifications via configured channels

## API Endpoints

- `GET /` - Home page
- `GET /upload` - Upload form
- `POST /upload` - Process uploaded images
- `GET /analyze` - Anomaly analysis form
- `POST /analyze` - Process anomaly analysis
- `GET /dashboard` - View analysis history
- `GET /api/analysis` - Get analysis data
- `GET /api/stats` - Get statistics

## Alert System

When illegal activities are detected, the system triggers:

1. **Immediate Web Alerts**: Sound + visual notifications
2. **Email Notifications**: Detailed reports sent to configured email
3. **SMS Alerts**: Critical alerts sent to mobile devices

## Security Notes

- Store credentials securely using environment variables
- Use App Passwords for Gmail instead of regular passwords
- Regularly rotate API keys and tokens
- Monitor notification usage to avoid service limits

## Production Deployment

For production use:

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up proper logging
3. Configure HTTPS
4. Use environment-specific configuration
5. Set up monitoring and alerting for the application itself

## License

This project is for educational and demonstration purposes.