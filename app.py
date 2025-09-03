import base64
import io
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, redirect, render_template, request, url_for, session, abort
from flask_cors import CORS

from database import db_session, init_db, Person, Attendance
from recognizer import FaceRecognizerService
from model import detect_faces_from_image
import cv2
import numpy as np
import shutil
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
CORS(app)

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

DATA_DIR = os.path.join(os.getcwd(), 'data')
FACES_DIR = os.path.join(DATA_DIR, 'faces')
MODEL_PATH = os.path.join(DATA_DIR, 'model.yml')

os.makedirs(FACES_DIR, exist_ok=True)

recognizer_service = FaceRecognizerService(dataset_dir=FACES_DIR, model_path=MODEL_PATH)


def _decode_base64_image_to_bgr(image_b64: str) -> np.ndarray:
    header_sep = ','
    if header_sep in image_b64:
        image_b64 = image_b64.split(header_sep, 1)[1]
    image_bytes = base64.b64decode(image_b64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_bgr


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


def is_admin_logged_in():
    return session.get('admin_logged_in', False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enroll')
def enroll_page():
    return render_template('enroll.html')


@app.route('/mark')
def mark_page():
    return render_template('mark.html')


@app.route('/logs')
def logs_page():
    # Get latest 200 attendance records
    attendances = db_session.query(Attendance).join(Person).order_by(Attendance.timestamp.desc()).limit(200).all()
    return render_template('logs.html', attendances=attendances)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password', '').strip()
        # Default admin password - change this in production!
        if password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('people_page'))
        else:
            return render_template('admin_login.html', error='Invalid password')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/people')
def people_page():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    people = Person.query.all()
    return render_template('people.html', people=people, is_admin=True)


@app.post('/api/enroll')
def api_enroll():
    payload = request.get_json(silent=True) or {}
    name = (payload.get('name') or '').strip()
    role = (payload.get('role') or 'student').strip().lower()
    if role not in ('student', 'teacher'):
        role = 'student'
    image_b64 = payload.get('image')
    raw_password = (payload.get('password') or '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name is required'}), 400
    if not image_b64:
        return jsonify({'ok': False, 'error': 'Image is required'}), 400
    if not raw_password or len(raw_password) < 4:
        return jsonify({'ok': False, 'error': 'Password is required (min 4 chars)'}), 400

    image_bgr = _decode_base64_image_to_bgr(image_b64)
    if image_bgr is None:
        return jsonify({'ok': False, 'error': 'Invalid image payload'}), 400

    faces = detect_faces_from_image(image_bgr)
    if not faces:
        return jsonify({'ok': False, 'error': 'No face detected'}), 400

    person = Person(name=name, role=role, password_hash=generate_password_hash(raw_password))
    db_session.add(person)
    db_session.commit()

    person_dir = os.path.join(FACES_DIR, f'person_{person.id}')
    os.makedirs(person_dir, exist_ok=True)

    # Save 5 augmentations (crops + slight variations)
    count_saved = 0
    for (x, y, w, h) in faces[:1]:
        face_img = image_bgr[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
        for idx, aug in enumerate([
            face_img,
            cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY),
            cv2.equalizeHist(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)),
            cv2.GaussianBlur(face_img, (3, 3), 0),
            cv2.bilateralFilter(face_img, 5, 75, 75)
        ]):
            if aug.ndim == 3:
                aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
            save_path = os.path.join(person_dir, f'{idx+1}.png')
            cv2.imwrite(save_path, aug)
            count_saved += 1

    recognizer_service.train_from_dataset()

    return jsonify({'ok': True, 'personId': person.id, 'saved': count_saved})


@app.post('/api/mark')
def api_mark():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get('image')
    if not image_b64:
        return jsonify({'ok': False, 'error': 'Image is required'}), 400

    image_bgr = _decode_base64_image_to_bgr(image_b64)
    if image_bgr is None:
        return jsonify({'ok': False, 'error': 'Invalid image payload'}), 400

    result = recognizer_service.predict(image_bgr)
    if not result['ok']:
        return jsonify(result), 400

    person_id = result['personId']
    person = Person.query.get(person_id)
    if person is None:
        return jsonify({'ok': False, 'error': 'Person not found'}), 404

    attendance = Attendance(person_id=person.id, timestamp=datetime.utcnow())
    db_session.add(attendance)
    db_session.commit()

    return jsonify({'ok': True, 'personId': person.id, 'name': person.name, 'confidence': result.get('confidence'), 'distance': result.get('distance')})


@app.post('/api/person/<int:person_id>/delete')
def api_delete_person(person_id: int):
    if not is_admin_logged_in():
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    person = Person.query.get(person_id)
    if person is None:
        return jsonify({'ok': False, 'error': 'Person not found'}), 404

    # Delete dataset folder
    person_dir = os.path.join(FACES_DIR, f'person_{person.id}')
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir, ignore_errors=True)

    # Delete attendance records and person
    db_session.query(Attendance).filter(Attendance.person_id == person.id).delete()
    db_session.delete(person)
    db_session.commit()

    # Retrain model after deletion
    recognizer_service.train_from_dataset()

    return jsonify({'ok': True})


@app.get('/api/recognizer/status')
def api_recognizer_status():
    """Get the current status of the face recognizer"""
    if not is_admin_logged_in():
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    
    status = recognizer_service.get_status()
    return jsonify({'ok': True, 'status': status})


@app.post('/api/recognizer/retrain')
def api_recognizer_retrain():
    """Force retrain the face recognizer model"""
    if not is_admin_logged_in():
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    
    success = recognizer_service.force_retrain()
    if success:
        return jsonify({'ok': True, 'message': 'Model retrained successfully'})
    else:
        return jsonify({'ok': False, 'error': 'Failed to retrain model'}), 500


def main():
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()