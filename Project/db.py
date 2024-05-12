import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {
                                  'databaseURL':"https://faceattendancepython-8f98b-default-rtdb.firebaseio.com/"
                              })
ref = db.reference("Students")

data = {
    "14036":{
        "name": "Avin Joshy",
        "major":"Btech",
        "department" : "CSE",
        "semester":"S6",
        "batch": "CSE-A",
        "roll_no":54,
        "admission_no": 14036,
        "attendance": 0,
        "total_hours":100,
        "standing": "G",
        "starting_year":2021,
        "last_attendance": "2024-05-09 12:00:00"
    },
"852741":{
        "name": "Emly Blunt",
        "major":"Btech",
        "department" : "CSE",
        "semester":"S6",
        "batch": "CSE-A",
        "roll_no":55,
        "admission_no": 14037,
        "attendance": 0,
        "total_hours":100,
        "standing": "G",
        "starting_year":2021,
        "last_attendance": "2024-05-09 12:00:00"
    },
"963852":{
        "name": "Elon Musk",
        "major":"Btech",
        "department" : "CSE",
        "semester":"S6",
        "batch": "CSE-A",
        "roll_no":56,
        "admission_no": 14038,
        "attendance": 0,
        "total_hours":100,
        "standing": "G",
        "starting_year":2021,
        "last_attendance": "2024-05-09 12:00:00"
    }
}

for key,value in data.items():
    ref.child(key).set(value)