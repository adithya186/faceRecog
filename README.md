# Face Recognition Attendance System

A comprehensive face-based attendance system using Flask, OpenCV (LBPH), SQLite, and modern web technologies. This system supports webcam enrollment, real-time attendance marking, and administrative management.

## Features

- 🎥 **Webcam-based enrollment** of new people with face detection
- 🧠 **LBPH face recognition** with automatic model training and retraining
- 📊 **Real-time attendance marking** with confidence scoring
- 👥 **Multi-user support** with proper model persistence
- 📋 **Attendance logs** with detailed history (latest 200 entries)
- 🔐 **Admin authentication** with password protection
- 🗄️ **SQLite database** with SQLAlchemy ORM
- 📱 **Responsive web interface** with modern UI/UX
- 🔧 **Admin panel** for user management and system monitoring

## Project Structure

```
faceRecog/
├── app.py                 # Main Flask application and API routes
├── database.py            # SQLAlchemy models and database initialization
├── recognizer.py          # LBPH face recognizer service
├── model.py              # Face detection helpers (Haar Cascade)
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── enroll.html
│   ├── mark.html
│   ├── logs.html
│   └── people.html
├── static/
│   └── css/
│       └── styles.css
├── data/                 # Runtime data (created automatically)
│   ├── faces/           # Face dataset (person_<id>/*.png)
│   └── model.yml        # Trained LBPH model
└── attendance.db        # SQLite database (created automatically)
```

## Requirements

- Python 3.8 or higher
- Webcam access
- Modern web browser with camera support

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adithya186/faceRecog.git
   cd faceRecog
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:5000`

## Usage

### 1. Enroll New Person
- Click "Enroll New Person"
- Enter a unique ID/name
- Select role (Student/Teacher)
- Set a password
- Click "Capture & Enroll" to save face samples and train the model

### 2. Mark Attendance
- Click "Mark Attendance"
- Look at the camera
- The system will automatically recognize and mark attendance
- You'll see your name and confidence score

### 3. View Logs
- Click "View Logs" to see recent attendance events
- Shows timestamp, person name, and confidence score

### 4. Admin Panel
- Click "Admin Login" and enter admin credentials
- Manage enrolled people
- View system status
- Force model retraining if needed

## Configuration

### Environment Variables

- `SECRET_KEY`: Flask secret key (default: auto-generated)
- `ATTENDANCE_DB_URL`: Database URL (default: `sqlite:///attendance.db`)
- `RECOG_UNKNOWN_DISTANCE`: Recognition threshold (default: 65)

### Recognition Threshold

The system uses LBPH (Local Binary Patterns Histograms) for face recognition. Lower distance values indicate better matches. You can adjust the threshold:

- **Lower threshold (e.g., 50)**: More strict recognition, fewer false positives
- **Higher threshold (e.g., 80)**: More lenient recognition, may accept similar faces

## API Endpoints

### Public Endpoints
- `GET /` - Home page
- `GET /enroll` - Enrollment page
- `GET /mark` - Attendance marking page
- `GET /logs` - Attendance logs page
- `POST /api/enroll` - Enroll new person
- `POST /api/mark` - Mark attendance

### Admin Endpoints
- `GET /admin/login` - Admin login page
- `POST /admin/login` - Admin authentication
- `GET /people` - People management page
- `GET /api/recognizer/status` - Get recognizer status
- `POST /api/recognizer/retrain` - Force model retraining
- `POST /api/person/<id>/delete` - Delete person

## Technical Details

### Face Recognition Pipeline

1. **Face Detection**: Uses Haar Cascade classifier to detect faces
2. **Image Preprocessing**: Converts to grayscale, applies histogram equalization
3. **Feature Extraction**: LBPH algorithm extracts local binary patterns
4. **Training**: Model trains on multiple augmented images per person
5. **Recognition**: Compares input face against trained model

### Data Augmentation

For each enrolled person, the system creates 5 variations:
- Original face crop
- Grayscale conversion
- Histogram equalized
- Gaussian blur
- Bilateral filter

This improves recognition accuracy and robustness.

### Model Persistence

- Face images stored in `data/faces/person_<id>/`
- Trained model saved as `data/model.yml`
- Model automatically loads on startup
- Automatic retraining if model loading fails

## Troubleshooting

### Common Issues

1. **"No face detected"**
   - Ensure good lighting
   - Position face clearly in camera view
   - Check camera permissions

2. **"Unknown face" for enrolled users**
   - Check if model was properly trained
   - Use admin panel to force retrain
   - Verify face images were saved correctly

3. **Camera access denied**
   - Grant camera permissions in browser
   - Check if another application is using the camera
   - Try refreshing the page

4. **Model not loading**
   - Check `data/model.yml` exists
   - Use admin panel to retrain model
   - Verify face dataset is not empty

### Debug Mode

Run with debug logging:
```bash
export FLASK_DEBUG=1
python app.py
```

## Production Deployment

### Using Waitress (Recommended)

```bash
pip install waitress
waitress-serve --port=5000 app:app
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables for Production

```bash
export SECRET_KEY="your-secure-secret-key"
export ATTENDANCE_DB_URL="postgresql://user:pass@localhost/attendance"
export RECOG_UNKNOWN_DISTANCE="60"
```

## Security Considerations

- Change default admin password
- Use HTTPS in production
- Regularly backup database and face data
- Implement rate limiting for API endpoints
- Consider data privacy regulations (GDPR, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## Changelog

### Version 1.0.0
- Initial release
- Face enrollment and recognition
- Attendance marking system
- Admin panel
- Multi-user support with proper model persistence
- Real-time recognition with confidence scoring