import os
import json
import re
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


genai.configure(api_key="apikeyyyy")


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Quiz generator route
@app.route('/quiz-generator')
def quiz_generator():
    return render_template('quiz-generator.html')

# YouTube insights route
@app.route('/youtube-insights')
def youtube_insights():
    return render_template('youtube-insights.html')

# File upload handler for quiz generation
@app.route('/upload-document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from the file (simplified, just for TXT files)
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # For other file types, we'd use appropriate libraries
            # like PyPDF2 for PDFs or python-docx for DOCX files
            text = f"Content extracted from {filename}"
        
        return jsonify({
            'success': True,
            'filename': filename,
            'text_preview': text[:500] + '...' if len(text) > 500 else text
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

# Generate quiz from document
@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    filename = data.get('filename')
    num_questions = data.get('num_questions', 10)
    difficulty = data.get('difficulty', 'medium')
    quiz_type = data.get('quiz_type', 'mcq')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Read file content
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        else:
            # For other file types, use appropriate libraries
            document_text = f"Content extracted from {filename}"
        
        # Generate quiz questions using Gemini
        try:
            quiz_questions = generate_quiz_from_content(document_text, num_questions, difficulty, quiz_type)
            
            # Check if we have a valid response
            if not quiz_questions or len(quiz_questions) == 0:
                logger.error("Empty quiz questions returned")
                return jsonify({'error': 'Failed to generate quiz questions. Please try again.'}), 500
                
            # Validate the structure of each question
            valid_questions = []
            for q in quiz_questions:
                if all(key in q for key in ['question', 'options', 'correctAnswer', 'difficulty']):
                    # Ensure correctAnswer is an integer and within valid range
                    try:
                        q['correctAnswer'] = int(q['correctAnswer'])
                        if q['correctAnswer'] < 0 or q['correctAnswer'] >= len(q['options']):
                            q['correctAnswer'] = 0  # Default to first option if out of range
                    except (ValueError, TypeError):
                        q['correctAnswer'] = 0  # Default to first option if not a valid number
                    
                    valid_questions.append(q)
            
            if len(valid_questions) == 0:
                logger.error("No valid questions in response")
                return jsonify({'error': 'Generated questions have invalid structure. Please try again.'}), 500
                
            return jsonify({'questions': valid_questions[:int(num_questions)]})
        except Exception as e:
            logger.error(f"Quiz generation error: {str(e)}")
            return jsonify({'error': f"Failed to generate quiz: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"File reading error: {str(e)}")
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

def generate_quiz_from_content(content, num_questions, difficulty, quiz_type):
    """
    Generate quiz questions using Gemini API with improved parsing
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Create a quiz with {num_questions} multiple-choice questions based on this content:

"{content[:4000]}"  // Limiting content for API constraints

Rules:
1. Difficulty level should be {difficulty}
2. Quiz type should be {quiz_type}
3. For multiple choice questions, provide 4 options with exactly one correct answer
4. Your ENTIRE response must be valid JSON with this exact structure:
[
  {{
    "question": "Question text goes here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correctAnswer": 2,  // Index of correct option (0-based)
    "difficulty": "{difficulty}"  // easy, medium, or hard
  }},
  // more questions...
]

IMPORTANT: Return ONLY the JSON array and nothing else. No explanations, no markdown, no codeblocks.
"""
        
        # Log the prompt for debugging
        logger.info(f"Sending prompt to Gemini for quiz generation")
        
        # Generate quiz
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Log response length for debugging
        logger.info(f"Received response of length {len(response_text)} characters")
        
        # Try to parse the entire response as JSON first
        try:
            quiz_questions = json.loads(response_text)
            if isinstance(quiz_questions, list) and len(quiz_questions) > 0:
                logger.info(f"Successfully parsed JSON directly, got {len(quiz_questions)} questions")
                return quiz_questions[:int(num_questions)]  # Limit to requested number
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed, trying alternative methods")
            
        # If that fails, try to extract JSON from the response
        try:
            # Look for JSON array pattern
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                quiz_questions = json.loads(json_str)
                if isinstance(quiz_questions, list) and len(quiz_questions) > 0:
                    logger.info(f"Extracted JSON using regex, got {len(quiz_questions)} questions")
                    return quiz_questions[:int(num_questions)]
        except Exception as e:
            logger.info(f"JSON extraction via regex failed: {str(e)}")
            
        # If all else fails, try to extract code block content
        try:
            # Look for markdown code blocks
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if code_block_match:
                json_str = code_block_match.group(1).strip()
                quiz_questions = json.loads(json_str)
                if isinstance(quiz_questions, list) and len(quiz_questions) > 0:
                    logger.info(f"Extracted JSON from code block, got {len(quiz_questions)} questions")
                    return quiz_questions[:int(num_questions)]
        except Exception as e:
            logger.info(f"JSON extraction from code block failed: {str(e)}")
            
        # Last resort: Try to manually clean the response and parse it
        try:
            # Remove common markdown and explanation text
            cleaned_text = re.sub(r'^[^[{]*', '', response_text)  # Remove text before first [ or {
            cleaned_text = re.sub(r'[^}\]]*$', '', cleaned_text)  # Remove text after last } or ]
            
            # Try to parse again
            quiz_questions = json.loads(cleaned_text)
            if isinstance(quiz_questions, list) and len(quiz_questions) > 0:
                logger.info(f"Parsed JSON after cleaning, got {len(quiz_questions)} questions")
                return quiz_questions[:int(num_questions)]
        except Exception as e:
            logger.info(f"Cleaned JSON parsing failed: {str(e)}")
        
        # If we still don't have a valid response, log the response for debugging
        logger.error(f"Failed to parse Gemini response into valid JSON. Response: {response_text[:200]}...")
        raise ValueError("Could not parse quiz questions from Gemini response")
    
    except Exception as e:
        logger.error(f"Error in quiz generation: {str(e)}")
        raise Exception(f"Error in quiz generation: {str(e)}")

# ### --- START OF UPDATED SECTION --- ###

def get_transcript(video_url):
    """
    Efficiently and robustly extracts transcript from a YouTube video.
    Handles multiple URL formats and common errors.
    """
    # Regex to find the video ID from various YouTube URL formats
    video_id_match = re.search(r"(?:v=|\/|youtu\.be\/|embed\/)([a-zA-Z0-9_-]{11})", video_url)
    
    if not video_id_match:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        return "Error: Invalid YouTube URL provided."

    video_id = video_id_match.group(1)

    try:
        # Fetch the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine the transcript text into a single string
        full_transcript = " ".join([entry['text'].strip() for entry in transcript_list])
        return full_transcript

    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video ID: {video_id}")
        return "Error: Transcripts are disabled for this video."
    
    except NoTranscriptFound:
        logger.error(f"No transcript found for video ID: {video_id}. It may not have captions.")
        return "Error: No transcript could be found for this video. Please try a different one."

    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching transcript for video ID {video_id}: {str(e)}")
        return f"Error: An unexpected error occurred. Please check the video URL and try again."

# ### --- END OF UPDATED SECTION --- ###

def generate_summary(transcript):
    """
    Generate summary using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash') # Changed model for consistency
        
        transcript = transcript.strip()
        
        if not transcript or len(transcript.split()) < 30:
            return "The transcript is too short to generate a meaningful summary."
        
        prompt = f"""Please provide a concise and comprehensive summary of the following transcript. 
        Focus on the key points, main ideas, and most important information. 
        The summary should be clear, coherent, and capture the essence of the content:

        {transcript}"""
        
        # Generate summary
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def translate_to_hindi(text):
    """
    Translate text to Hindi using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare prompt for translation
        prompt = f"""Translate the following English text to Hindi. 
        Ensure the translation is natural, fluent, and maintains the original meaning:

        {text}"""
        
        # Generate translation
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"Error translating text: {str(e)}"

def answer_question(transcript, question):
    """
    Answer questions using Gemini API
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare prompt for question answering
        prompt = f"""Based on the following transcript, please answer the question as accurately and concisely as possible:

Transcript:
{transcript}

Question: {question}

Answer:"""
        
        # Generate answer
        response = model.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error answering question: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Summarization route using Gemini
    """
    data = request.json
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    # Fetch transcript
    transcript = get_transcript(video_url)
    
    if "Error:" in transcript:
        return jsonify({"error": transcript}), 400
    
    # Generate summary
    summary = generate_summary(transcript)
    
    if "Error:" in summary:
        return jsonify({"error": summary}), 500
    
    return jsonify({"summary": summary})

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translation route for converting English to Hindi
    """
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided for translation"}), 400
    
    # Translate text to Hindi
    hindi_text = translate_to_hindi(text)
    
    if "Error:" in hindi_text:
        return jsonify({"error": hindi_text}), 500
    
    return jsonify({"translated_text": hindi_text})

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Question answering route using Gemini
    """
    data = request.json
    video_url = data.get('video_url')
    question = data.get('question')
    
    if not video_url or not question:
        return jsonify({"error": "Video URL or question not provided"}), 400
    
    # Fetch transcript
    transcript = get_transcript(video_url)
    
    if "Error:" in transcript:
        return jsonify({"error": transcript}), 400
    
    # Answer question
    answer = answer_question(transcript, question)
    
    if "Error:" in answer:
        return jsonify({"error": answer}), 500
    
    return jsonify({"answer": answer})

# Performance optimization for Flask
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

if __name__ == "__main__":
    # Use a production WSGI server in production
    app.run(debug=True, threaded=True)
