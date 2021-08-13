from datetime import datetime
from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import folium
import os
import random

from forms import NavigationForm
import visualize

app = Flask(__name__)

sess = Session()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SESSION_TYPE'] = 'filesystem'

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# path_manhattan = os.path.join(parent_dir, "pathData/manhattan_geo_paths.gpkg")
path_manhattan = "/home/nlp/tzufar/Pycharm/second_year/cabby-tmp4/spatial_samples.gpkg"
osm_maps_instructions_manhattan = visualize.get_maps_and_instructions(path_manhattan)
size_dataset = len(osm_maps_instructions_manhattan)

db = SQLAlchemy(app)

N_TASKS_PER_USER = 5


class Instruction(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  sample_number = db.Column(db.Integer, nullable=False)
  content = db.Column(db.Text, nullable=False)
  date_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  date_finish = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

  def __repr__(self):
    return f"Path Descriptions('{self.id}', '{self.sample_number}', '{self.date_finish}')"


@app.route("/")
@app.route("/home")
def home():
  session["task"] = 0
  return redirect(url_for('index'))


@app.route("/index")
def index():
  session["task"] += 1
  if session["task"] <= N_TASKS_PER_USER:
    sample_number = random.randint(0, size_dataset)
    return redirect(url_for('task', sample=sample_number))
  else:
    session.pop('_flashes', None)
    return render_template('end.html', bar=100)


@app.route("/task/<sample>", methods=['GET', 'POST'])
def task(sample):
  sample = int(sample)
  folium_map, instruction, landmarks = osm_maps_instructions_manhattan[sample]

  folium_map.save('templates/map.html')
  form = NavigationForm()
  form.landmarks = landmarks

  if form.validate_on_submit():
    content = request.form['content']

    path = Instruction(sample_number=sample, content=content)
    db.session.add(path)
    db.session.commit()

    flash(session["task"], 'success')
    return redirect(url_for('index'))

  if session["task"] == 0:
    task = 0
  else:
    task = session["task"] - 1
  progress_task = round(task / N_TASKS_PER_USER * 100)
  return render_template('task.html',
                         form=form,
                         bar=progress_task,
                         title=session["task"],
                         )

  # return folium_map._repr_html_()


@app.route('/map')
def map():
  return render_template('map.html')


if __name__ == '__main__':
  sess = Session()
  app.run(host='0.0.0.0', port=5000, debug=False)
