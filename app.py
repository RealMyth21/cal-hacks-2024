from flask import Flask, request, url_for, redirect, render_template
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

#app.config['UPLOAD_FOLDER'] = 'path/to/upload_folder' 
app.config['UPLOAD_FOLDER'] = 'cal-hacks-2024\static\css\audios'
app.config['DEBUG'] = True

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    try:
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                print("No file part in request")
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an empty file without a filename
            if file.filename == '':
                print("No selected file")
                return redirect(request.url)
            if file:
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(f"File saved: {filename}")
                return redirect(url_for('demo'))

        return render_template('demo.html')
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the file upload process."


"""
@app.route("/contact")
def contact():
    return render_template("contact.html")
"""

if __name__ == "__main__":
    app.run(debug=True)
