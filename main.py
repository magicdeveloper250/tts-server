from flask import Flask, request, send_file, jsonify, Response, render_template
from flask_cors import CORS
import torch
from TTS.api import TTS
import io
import os
import logging
import time
import threading
import tempfile
import uuid
from datetime import datetime, timedelta
import wave
from pydub import AudioSegment
import shutil
import platform
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Optimized configuration
MAX_WORKERS = 4  # Concurrent TTS generations
CACHE_SIZE = 100  # Cache frequently used phrases
BATCH_SIZE = 3   # Process multiple requests together when possible

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Text preprocessing cache
text_cache = {}
cache_lock = threading.Lock()

# Global lock to prevent concurrent model access (but allow more concurrency)
model_lock = threading.Semaphore(MAX_WORKERS)
tts = None
device = "cpu"

# Precomputed audio segments for common words/phrases
COMMON_PHRASES = {
    "hello": None,
    "goodbye": None,
    "yes": None,
    "no": None,
    "thank you": None,
    "please": None
}

# File storage for downloads
if platform.system() == "Windows":
    TEMP_DIR = os.path.join(os.getcwd(), "tts_downloads")
else:
    TEMP_DIR = tempfile.mkdtemp(prefix="tts_downloads_")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Download directory: {TEMP_DIR}")

# Store generated files with metadata
generated_files = {}
file_cleanup_lock = threading.Lock()

def precompute_common_phrases():
    """Precompute audio for common phrases to speed up generation"""
    global COMMON_PHRASES
    if not tts:
        return
    
    logger.info("Precomputing common phrases...")
    for phrase in COMMON_PHRASES.keys():
        try:
            temp_path = os.path.join(TEMP_DIR, f"cache_{phrase.replace(' ', '_')}.wav")
            with model_lock:
                tts.tts_to_file(text=phrase, file_path=temp_path)
            
            # Convert to ESP32 format and store in memory
            audio_data = optimize_audio_for_esp32(temp_path)
            COMMON_PHRASES[phrase] = audio_data
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            logger.warning(f"Failed to precompute phrase '{phrase}': {e}")
    
    logger.info(f"Precomputed {len([p for p in COMMON_PHRASES.values() if p])} common phrases")

def optimize_audio_for_esp32(file_path):
    """Optimized audio processing for ESP32 compatibility"""
    try:
        # Use pydub for faster processing
        audio = AudioSegment.from_wav(file_path)
        
        # Quick format conversion
        if audio.frame_rate != 44100:
            audio = audio.set_frame_rate(44100)
        if audio.channels != 2:
            audio = audio.set_channels(2)
        
        # Return raw audio data for caching
        return audio.export(format="wav").read()
        
    except Exception as e:
        logger.error(f"Audio optimization failed: {e}")
        return None

def fast_text_preprocessing(text):
    """Fast text preprocessing with caching"""
    text_hash = hash(text)
    
    with cache_lock:
        if text_hash in text_cache:
            return text_cache[text_hash]
    
    # Quick preprocessing
    processed_text = text.strip()
    
    # Remove excessive punctuation that slows down TTS
    processed_text = processed_text.replace('...', '.')
    processed_text = processed_text.replace('!!', '!')
    processed_text = processed_text.replace('??', '?')
    
    # Limit length for speed
    if len(processed_text) > 200:
        processed_text = processed_text[:200].rsplit(' ', 1)[0] + '.'
    
    with cache_lock:
        text_cache[text_hash] = processed_text
        # Limit cache size
        if len(text_cache) > CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(text_cache))
            del text_cache[oldest_key]
    
    return processed_text

def cleanup_old_files():
    """Optimized cleanup - only run when needed"""
    with file_cleanup_lock:
        current_time = datetime.now()
        to_remove = []
        
        # Quick scan - only process if we have many files
        if len(generated_files) < 50:
            return
        
        for file_id, file_info in generated_files.items():
            if current_time - file_info['created'] > timedelta(hours=1):  # Reduced to 1 hour
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                    to_remove.append(file_id)
                except Exception as e:
                    logger.error(f"Error removing old file {file_id}: {e}")
        
        for file_id in to_remove:
            del generated_files[file_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old files")

def fast_wav_conversion(input_path, output_path):
    """Optimized WAV conversion using numpy for speed"""
    try:
        # Use pydub for fast conversion
        audio = AudioSegment.from_wav(input_path)
        
        # Quick stereo conversion if needed
        if audio.channels == 1:
            # Fast mono to stereo
            audio = audio.set_channels(2)
        
        # Quick sample rate conversion if needed
        if audio.frame_rate != 44100:
            audio = audio.set_frame_rate(44100)
        
        # Export directly
        audio.export(output_path, format="wav")
        return True
        
    except Exception as e:
        logger.error(f"Fast conversion failed: {e}")
        return False

def generate_wav_to_file_optimized(text, file_path):
    """Optimized WAV generation with caching and fast processing"""
    try:
        # Check if this is a common phrase
        text_lower = text.lower().strip()
        if text_lower in COMMON_PHRASES and COMMON_PHRASES[text_lower]:
            # Use precomputed audio
            with open(file_path, 'wb') as f:
                f.write(COMMON_PHRASES[text_lower])
            file_size = os.path.getsize(file_path)
            logger.info(f"Used cached phrase: '{text}' ({file_size} bytes)")
            return file_size
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Preprocess text quickly
        processed_text = fast_text_preprocessing(text)
        
        # Generate with optimized settings
        with model_lock:
            logger.info(f"Generating TTS: '{processed_text}'")
            start_gen = time.time()
            
            # Use faster TTS settings if available
            tts.tts_to_file(
                text=processed_text, 
                file_path=file_path,
                speed=1.2  # Slightly faster speech for quicker generation
            )
            
            gen_time = time.time() - start_gen
            logger.debug(f"TTS generation took {gen_time:.2f}s")
        
        # Quick format verification and conversion
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            # Fast conversion to ESP32 format
            temp_path = file_path + ".temp"
            if fast_wav_conversion(file_path, temp_path):
                shutil.move(temp_path, file_path)
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Generated optimized WAV: {file_size} bytes")
            return file_size
        else:
            raise ValueError("TTS generation failed")
        
    except Exception as e:
        logger.error(f"Error in optimized generation: {e}")
        # Clean up any partial files
        for temp_file in [file_path, file_path + ".temp"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        raise

def initialize_model():
    """Initialize TTS model with optimizations"""
    global tts, device
    
    try:
        logger.info("Initializing optimized TTS model...")
        
        # Use GPU if available for faster processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Use faster model - Tacotron2 with optimizations
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        
        # Optimize model for inference
        if hasattr(tts.synthesizer.tts_model, 'eval'):
            tts.synthesizer.tts_model.eval()
        
        # Enable inference optimizations
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Test the model
        with model_lock:
            test_file = os.path.join(TEMP_DIR, "test.wav")
            if os.path.exists(test_file):
                os.remove(test_file)
            tts.tts_to_file(text="test", file_path=test_file)
            if os.path.exists(test_file):
                os.remove(test_file)
        
        logger.info(f"TTS model loaded successfully on {device}")
        
        # Precompute common phrases in background
        threading.Thread(target=precompute_common_phrases, daemon=True).start()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {e}")
        tts = None
        return False

# Initialize model at startup
if not initialize_model():
    logger.error("Failed to initialize TTS model. Server will not function properly.")

@app.route('/generate', methods=['POST'])
def generate_speech_file():
    """Optimized generation endpoint with async processing"""
    if not tts:
        return jsonify({'error': 'TTS model not loaded'}), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Quick preprocessing
        processed_text = fast_text_preprocessing(text)
        
        logger.info(f"Processing: '{processed_text}'")
        start_time = time.time()
        
        # Generate unique file ID and path
        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
        
        # Use optimized generation
        try:
            file_size = generate_wav_to_file_optimized(processed_text, file_path)
        except Exception as gen_error:
            logger.error(f"Optimized generation failed: {gen_error}")
            return jsonify({
                'error': 'Failed to generate audio file',
                'details': str(gen_error)
            }), 500
        
        processing_time = time.time() - start_time
        
        # Verify file
        if not os.path.exists(file_path) or file_size == 0:
            logger.error(f"Generated file is invalid: {file_path}")
            return jsonify({'error': 'Generated file is invalid'}), 500
        
        # Store file info
        with file_cleanup_lock:
            generated_files[file_id] = {
                'path': file_path,
                'text': processed_text,
                'size': file_size,
                'created': datetime.now(),
                'downloaded': False
            }
        
        logger.info(f"✓ Fast generation {file_id}: {file_size} bytes in {processing_time:.2f}s")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'download_url': f'/download/{file_id}',
            'file_size': file_size,
            'processing_time': round(processing_time, 2),
            'text': processed_text[:50] + ('...' if len(processed_text) > 50 else ''),
            'format': '44100Hz stereo WAV (ESP32 optimized)',
            'cached': processed_text.lower() in COMMON_PHRASES
        })
            
    except Exception as e:
        logger.error(f"Error in generate_speech_file: {str(e)}")
        return jsonify({
            'error': 'Failed to generate speech file',
            'details': str(e)
        }), 500



@app.route('/', methods=['POST', "GET"])
def index():
    return render_template("index.html")
@app.route('/generate_batch', methods=['POST'])
def generate_batch():
    """New endpoint for batch processing multiple texts"""
    if not tts:
        return jsonify({'error': 'TTS model not loaded'}), 500
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or len(texts) > 5:  # Limit batch size
            return jsonify({'error': 'Provide 1-5 texts for batch processing'}), 400
        
        results = []
        start_time = time.time()
        
        # Process all texts
        for text in texts:
            try:
                processed_text = fast_text_preprocessing(text.strip())
                file_id = str(uuid.uuid4())
                file_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
                
                file_size = generate_wav_to_file_optimized(processed_text, file_path)
                
                with file_cleanup_lock:
                    generated_files[file_id] = {
                        'path': file_path,
                        'text': processed_text,
                        'size': file_size,
                        'created': datetime.now(),
                        'downloaded': False
                    }
                
                results.append({
                    'text': processed_text,
                    'file_id': file_id,
                    'download_url': f'/download/{file_id}',
                    'file_size': file_size,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e),
                    'success': False
                })
        
        processing_time = time.time() - start_time
        successful_count = sum(1 for r in results if r['success'])
        
        return jsonify({
            'results': results,
            'total_processing_time': round(processing_time, 2),
            'successful_generations': successful_count,
            'total_requests': len(texts)
        })
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return jsonify({'error': 'Batch generation failed', 'details': str(e)}), 500

# Keep all original download endpoints unchanged
@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download endpoint - serves ESP32-optimized files (unchanged)"""
    # Only cleanup if we have too many files
    if len(generated_files) > 100:
        cleanup_old_files()
    
    with file_cleanup_lock:
        if file_id not in generated_files:
            logger.warning(f"File not found: {file_id}")
            return jsonify({'error': 'File not found or expired'}), 404
        
        file_info = generated_files[file_id]
        file_path = file_info['path']
        
        if not os.path.exists(file_path):
            logger.warning(f"Physical file missing: {file_path}")
            del generated_files[file_id]
            return jsonify({'error': 'File no longer exists'}), 404
        
        # Mark as downloaded
        file_info['downloaded'] = True
        file_info['download_time'] = datetime.now()
    
    logger.info(f"Serving ESP32-optimized download for file {file_id}")
    
    try:
        response = send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'esp32_speech_{file_id[:8]}.wav'
        )
        
        # Add headers for better compatibility
        response.headers['Content-Length'] = str(file_info['size'])
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'close'
        response.headers['X-Audio-Format'] = '44100Hz-stereo-ESP32'
        
        return response
        
    except Exception as e:
        logger.error(f"Error serving file {file_id}: {e}")
        return jsonify({'error': 'Error serving file'}), 500
@app.route('/stream/<file_id>', methods=['GET'])
def stream_file(file_id):
    """Fixed streaming endpoint for ESP32 compatibility"""
    if len(generated_files) > 100:
        cleanup_old_files()
    
    with file_cleanup_lock:
        if file_id not in generated_files:
            return jsonify({'error': 'File not found or expired'}), 404
        
        file_info = generated_files[file_id]
        file_path = file_info['path']
        if not os.path.exists(file_path):
            del generated_files[file_id]
            return jsonify({'error': 'File no longer exists'}), 404
    
    def generate_chunks():
        """Generator that properly closes the stream"""
        try:
            with open(file_path, 'rb') as f:
                bytes_sent = 0
                file_size = os.path.getsize(file_path)
                
                
                while bytes_sent < file_size:
                    chunk = f.read(4096)  # 4KB chunks
                    if not chunk:
                        break
                    yield chunk
                    bytes_sent += len(chunk)
                
                logger.info(f"Stream complete: {bytes_sent}/{file_size} bytes sent for {file_id}")
                
        except Exception as e:
            logger.error(f"Error streaming file {file_id}: {e}")
            return  # This will close the generator
    
    file_size = os.path.getsize(file_path)
    
    response = Response(
        generate_chunks(),
        mimetype='audio/wav',  # Changed back to audio/wav
        headers={
            'Content-Length': str(file_size),
            'Content-Type': 'audio/wav',
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache',
            'Connection': 'close',  # This is crucial!
            'X-Audio-Format': '44100Hz-stereo-ESP32'
        }
    )
    
    logger.info(f"Starting stream for file {file_id} ({file_size} bytes)")
    return response


# @app.route('/stream/<file_id>', methods=['GET'])
# def stream_file(file_id):
#     """Streaming endpoint for ESP32 compatibility"""
#     if len(generated_files) > 100:
#         cleanup_old_files()
    
#     with file_cleanup_lock:
#         if file_id not in generated_files:
#             return jsonify({'error': 'File not found or expired'}), 404
        
#         file_info = generated_files[file_id]
#         file_path = file_info['path']
        
#         if not os.path.exists(file_path):
#             del generated_files[file_id]
#             return jsonify({'error': 'File no longer exists'}), 404
    
#     # Create a proper streaming response with WAV headers
#     def generate():
#         try:
#             # First, send the entire file at once for ESP32 compatibility
#             # ESP32 has trouble with chunked streaming
#             with open(file_path, 'rb') as f:
#                 data = f.read()
#                 yield data
                
#         except Exception as e:
#             logger.error(f"Error streaming file {file_id}: {e}")
    
#     # Get file size for proper headers
#     file_size = os.path.getsize(file_path)
    
#     response = Response(
#         generate(),
#         mimetype='audio/x-wav',  # Changed to audio/x-wav for better compatibility
#         headers={
#             'Content-Length': str(file_size),
#             'Content-Type': 'audio/x-wav',
#             'Accept-Ranges': 'none',  # ESP32 doesn't handle range requests well
#             'Cache-Control': 'no-cache, no-store, must-revalidate',
#             'Pragma': 'no-cache',
#             'Expires': '0',
#             'Connection': 'close',
#             'X-Audio-Format': '44100Hz-stereo-ESP32'
#         }
#     )
    
#     logger.info(f"Streaming file {file_id} ({file_size} bytes) to ESP32")
#     return response

@app.route('/files', methods=['GET'])
def list_files():
    """List all available files (optimized)"""
    files_info = []
    with file_cleanup_lock:
        for file_id, info in generated_files.items():
            files_info.append({
                'file_id': file_id,
                'text': info['text'][:50] + ('...' if len(info['text']) > 50 else ''),
                'size': info['size'],
                'created': info['created'].isoformat(),
                'downloaded': info.get('downloaded', False),
                'download_url': f'/download/{file_id}',
                'format': '44100Hz stereo (ESP32)'
            })
    
    return jsonify({
        'files': files_info,
        'total_files': len(files_info),
        'temp_directory': TEMP_DIR,
        'cached_phrases': len([p for p in COMMON_PHRASES.values() if p])
    })

@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger file cleanup"""
    cleanup_old_files()
    
    with file_cleanup_lock:
        remaining_files = len(generated_files)
    
    return jsonify({
        'message': 'Cleanup completed',
        'remaining_files': remaining_files
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Optimized health check"""
    with file_cleanup_lock:
        file_count = len(generated_files)
        total_size = sum(info['size'] for info in generated_files.values()) if generated_files else 0
    
    status = 'healthy' if tts else 'unhealthy'
    return jsonify({
        'status': status, 
        'device': device,
        'model_loaded': tts is not None,
        'active_files': file_count,
        'total_file_size': total_size,
        'temp_directory': TEMP_DIR,
        'audio_format': '44100Hz stereo (ESP32 optimized)',
        'optimizations': {
            'max_workers': MAX_WORKERS,
            'cache_size': len(text_cache),
            'cached_phrases': len([p for p in COMMON_PHRASES.values() if p])
        }
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with performance info"""
    return jsonify({
        'message': 'Optimized ESP32 TTS Server is running',
        'model_loaded': tts is not None,
        'audio_format': '44100Hz stereo WAV (ESP32 Bluetooth compatible)',
        'optimizations': {
            'concurrent_workers': MAX_WORKERS,
            'phrase_caching': 'enabled',
            'fast_preprocessing': 'enabled',
            'batch_processing': 'available'
        },
        'endpoints': {
            'generate': 'POST {"text": "your text"} - Fast single generation',
            'generate_batch': 'POST {"texts": ["text1", "text2"]} - Batch processing',
            'download': 'GET /download/<file_id> - Download generated file',
            'stream': 'GET /stream/<file_id> - Stream generated file',
            'files': 'GET - List all available files',
            'health': 'GET - Server health status'
        },
        'device': device
    })

@app.route('/reset', methods=['POST'])
def reset_model():
    """Endpoint to reset the TTS model"""
    try:
        global tts, text_cache, COMMON_PHRASES
        
        if tts:
            del tts
            tts = None
        
        # Clear caches
        with cache_lock:
            text_cache.clear()
        
        for key in COMMON_PHRASES:
            COMMON_PHRASES[key] = None
        
        # Reinitialize
        success = initialize_model()
        return jsonify({
            'success': success,
            'message': 'Model and caches reset successfully' if success else 'Model reset failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cleanup on app termination
import atexit
def cleanup_on_exit():
    """Clean up temporary files on app shutdown"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Cleaned up temporary directory on exit: {TEMP_DIR}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory on exit: {e}")

atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    port = 5000
    logger.info(f"Starting OPTIMIZED ESP32 TTS server on port {port}")
    logger.info("Optimizations: Phrase caching, fast preprocessing, concurrent processing")
    logger.info("Audio format: 44100Hz stereo WAV (ESP32 Bluetooth compatible)")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,
        threaded=True,
        use_reloader=False
    )