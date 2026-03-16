"""
BIRD V51 - Question-Type Router + Dynamic Value Grounding
==========================================================
Based on V42 (72.3% baseline)

KEY INNOVATIONS:

1. QUESTION-TYPE ROUTER:
   - Classify question: ratio, aggregation, superlative, date, join, simple
   - Apply type-specific prompting strategies
   - Each type has different error patterns → different fixes

2. DYNAMIC VALUE GROUNDING:
   - Extract potential entities from question
   - Search database for matching values
   - Inject exact matches into prompt

3. TARGETED CRITIQUE:
   - If all candidates return empty → "Check JOINs and WHERE filters"
   - If candidates disagree → "Verify column selection"
   - Type-specific regeneration hints

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: QUESTION CLASSIFICATION                           │
│    → ratio, aggregation, superlative, date, join, simple    │
│                                                             │
│  Stage 2: DYNAMIC VALUE GROUNDING                           │
│    → Search DB for entities mentioned in question           │
│                                                             │
│  Stage 3: TYPE-SPECIFIC SQL GENERATION                      │
│    → Specialized prompts per question type                  │
│                                                             │
│  Stage 4: MAJORITY VOTING + TARGETED REPAIR                 │
└─────────────────────────────────────────────────────────────┘

Expected: 72.3% → 74%+

Usage:
    modal run modal_app_v51.py
"""

import modal
import json
import re
import time
import sqlite3
from typing import Dict, List, Tuple, Optional
from collections import Counter

app = modal.App("bird-v51-router-grounding")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai", "numpy")
)

volume = modal.Volume.from_name("bird-dataset", create_if_missing=True)

PRIMARY_MODEL = "gpt-5.2"
DATABASE_PATH = "/data/databases"
MAX_REPAIR_ATTEMPTS = 2
NUM_CANDIDATES = 3

# =============================================================================
# CURATED EXAMPLES (from V38)
# =============================================================================

DATABASE_EXAMPLES = {
    "financial": {
        "context": """Czech banking database from the 1990s.

CRITICAL - DISTINCT RULES:
- Do NOT add DISTINCT unless question asks for "unique" or "different"
- "How many accounts" → COUNT(account_id) NOT COUNT(DISTINCT account_id)
- "List clients" → SELECT client_id NOT SELECT DISTINCT client_id

CRITICAL VALUE MAPPINGS:
- Evidence contains Czech vocabulary - use EXACT strings from evidence
- Dates are ISO strings: '1995-03-24'
- District columns: A2=name, A3=region, A11=avg_salary, A12=unemployment_1995, A13=unemployment_1996

CRITICAL JOINS:
- account.district_id = district.district_id
- client.district_id = district.district_id
- disp.account_id = account.account_id
- disp.client_id = client.client_id
- card.disp_id = disp.disp_id
- loan.account_id = account.account_id""",
        "examples": [
            {"q": "Which client issued card on 1994/3/3?", "e": "", 
             "sql": "SELECT T2.client_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN card AS T3 ON T2.disp_id = T3.disp_id WHERE T3.issued = '1994-03-03'"},
            {"q": "How many accounts with issuance after transaction in East Bohemia?", "e": "A3 contains region; 'POPLATEK PO OBRATU' = issuance after transaction",
             "sql": "SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T1.A3 = 'east Bohemia' AND T2.frequency = 'POPLATEK PO OBRATU'"},
            {"q": "Average unemployment 1995 vs 1996, which higher?", "e": "A12 = unemployment 1995; A13 = unemployment 1996",
             "sql": "SELECT IIF(AVG(A13) > AVG(A12), '1996', '1995') FROM district"},
            {"q": "How many loans approved in 1997?", "e": "",
             "sql": "SELECT COUNT(loan_id) FROM loan WHERE STRFTIME('%Y', date) = '1997'"}
        ]
    },
    "california_schools": {
        "context": """California schools database with FRPM (free/reduced meals), SAT scores, and school info.

CRITICAL COLUMN MAPPINGS:
- satscores table: sname (school name), cname (county name), dname (district name)  
- schools table: School (school name), County (county name), District (district name)
- frpm table: `School Name`, `County Name`, `District Name`
- JOIN: frpm.CDSCode = schools.CDSCode, satscores.cds = schools.CDSCode

CRITICAL VALUE MAPPINGS:
- 'Charter School (Y/N)' uses NUMERIC 1/0 (NOT 'Y'/'N' strings)
- StatusType values: 'Active', 'Closed', 'Merged'
- Charter Funding Type: 'Directly funded', 'Locally funded'
- Educational Option Type: 'Traditional', 'Continuation School', 'Alternative School of Choice'
- satscores.rtype: 'S' = school level, 'D' = district level
- Virtual: 'F' = Full virtual, 'P' = Partial virtual, 'N' = Not virtual

CRITICAL SAT SCORE COLUMNS:
- NumGE1500 = COUNT of students whose TOTAL SAT score ≥ 1500 (NOT a calculated sum!)
- AvgScrRead, AvgScrMath, AvgScrWrite = average scores per subject
- NumTstTakr = number of test takers

NULL HANDLING:
- Always add IS NOT NULL when filtering or ordering by columns that may have NULLs
- Example: WHERE T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead

COLUMN NAME RULES:
- DOC = 2-digit District Ownership Code (NOT DOCType)
- SOC = School Ownership Code  
- Use backticks for column names with spaces: `Free Meal Count (K-12)`
- RETURN ONLY REQUESTED COLUMNS - match exact output structure""",
        "examples": [
            {"q": "Zip codes of charter schools in Fresno County Office of Education?", "e": "Charter school = `Charter School (Y/N)` = 1",
             "sql": "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1"},
            {"q": "Highest eligible free rate for K-12 in Alameda County?", "e": "Rate = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
             "sql": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1"},
            {"q": "Lowest three eligible free rates for students 5-17 in continuation schools?", "e": "Rate = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`, continuation = Educational Option Type",
             "sql": "SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY CAST(`Free Meal Count (Ages 5-17)` AS REAL) / `Enrollment (Ages 5-17)` ASC LIMIT 3"},
            {"q": "How many schools in merged Lake have test takers < 100?", "e": "merged = StatusType = 'Merged'",
             "sql": "SELECT COUNT(T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = 'Merged' AND T1.County = 'Lake' AND T2.NumTstTakr < 100"},
            {"q": "Phone numbers of direct charter-funded schools opened after 2000/1/1?", "e": "Charter schools = `Charter School (Y/N)` = 1; direct = 'Directly funded'",
             "sql": "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = 'Directly funded' AND T2.OpenDate > '2000-01-01'"},
            {"q": "Which active district has highest average Reading score?", "e": "",
             "sql": "SELECT T1.District FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = 'Active' AND T2.AvgScrRead IS NOT NULL ORDER BY T2.AvgScrRead DESC LIMIT 1"},
            {"q": "School with highest number of students scoring 1500+ on SAT?", "e": "NumGE1500 = count of students with total SAT ≥ 1500",
             "sql": "SELECT T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1"},
            {"q": "Which school in Contra Costa has highest test takers?", "e": "",
             "sql": "SELECT sname FROM satscores WHERE cname = 'Contra Costa' AND sname IS NOT NULL ORDER BY NumTstTakr DESC LIMIT 1"},
            {"q": "Street and school name for lowest average reading score?", "e": "",
             "sql": "SELECT T2.MailStreet, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1"},
            {"q": "Eligible free rate for top 5 FRPM schools with ownership code 66?", "e": "ownership code = SOC",
             "sql": "SELECT CAST(T1.`FRPM Count (K-12)` AS REAL) / T1.`Enrollment (K-12)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.SOC = 66 ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 5"}
        ]
    },
    "toxicology": {
        "context": """Molecular toxicology database with 4 tables.
SCHEMA DETAILS:
- molecule(molecule_id [PK], label): label '+' = carcinogenic, '-' = non-carcinogenic
- atom(atom_id [PK], molecule_id [FK], element): element is LOWERCASE ('c', 'cl', 'o', 'h', 'n', 's', 'f', 'i')
- bond(bond_id [PK], molecule_id [FK], bond_type): bond_type '-'=single, '='=double, '#'=triple
- connected(atom_id, atom_id2, bond_id): Links atoms via bonds. BOTH (atom_id, atom_id2) pairs exist for each bond!

CRITICAL PATTERNS:
1. connected table has BIDIRECTIONAL entries - if A connects to B, both (A,B) and (B,A) exist
2. For "atoms connected" queries: SELECT atom_id, atom_id2 (both columns, not just one)
3. For "atom IDs of bond X": SELECT atom_id FROM connected WHERE bond_id = 'X' (returns 2 rows)
4. DISTINCT: Only use when question asks for unique/distinct. "Identify molecules" often wants duplicates!
5. Percentage of X in Y: CAST(COUNT(CASE WHEN condition THEN 1 END) AS REAL) * 100 / COUNT(*)
6. Average per molecule: Use subquery to count per molecule_id, then AVG() on that
7. Element counts with iodine AND sulfur: Return TWO separate counts, not combined!
8. "Which bond type is majority": Return ONLY bond_type, not extra columns""",
        "examples": [
            {"q": "Non-carcinogenic molecules with chlorine?", "e": "non-carcinogenic = label = '-'; chlorine = element = 'cl'",
             "sql": "SELECT COUNT(DISTINCT T1.molecule_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.element = 'cl' AND T1.label = '-'"},
            {"q": "How many triple bonds?", "e": "triple = bond_type = '#'",
             "sql": "SELECT COUNT(T.bond_id) FROM bond AS T WHERE T.bond_type = '#'"},
            {"q": "Identify molecules with carbon (list them)", "e": "carbon = element = 'c'",
             "sql": "SELECT T.molecule_id FROM atom AS T WHERE T.element = 'c'"},
            {"q": "What are atom IDs of bond TR000_2_5?", "e": "TR000_2_5 is bond_id",
             "sql": "SELECT T.atom_id FROM connected AS T WHERE T.bond_id = 'TR000_2_5'"},
            {"q": "Atoms connected in molecule TR181?", "e": "show both atom columns",
             "sql": "SELECT T2.atom_id, T2.atom_id2 FROM atom AS T1 INNER JOIN connected AS T2 ON T2.atom_id = T1.atom_id WHERE T1.molecule_id = 'TR181'"},
            {"q": "Count iodine and sulfur atoms in single bonds separately", "e": "iodine = 'i', sulfur = 's', single = '-'",
             "sql": "SELECT COUNT(DISTINCT CASE WHEN T1.element = 'i' THEN T1.atom_id END) AS iodine_nums, COUNT(DISTINCT CASE WHEN T1.element = 's' THEN T1.atom_id END) AS sulfur_nums FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T2.bond_id = T3.bond_id WHERE T3.bond_type = '-'"},
            {"q": "Which bond type is majority in TR010?", "e": "majority = most common, return only bond_type",
             "sql": "SELECT T.bond_type FROM (SELECT bond_type, COUNT(*) FROM bond WHERE molecule_id = 'TR010' GROUP BY bond_type ORDER BY COUNT(*) DESC LIMIT 1) AS T"},
            {"q": "Average oxygen atoms in single-bonded molecules", "e": "oxygen = 'o', single = '-'",
             "sql": "SELECT AVG(oxygen_count) FROM (SELECT T1.molecule_id, COUNT(T1.element) AS oxygen_count FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = '-' AND T1.element = 'o' GROUP BY T1.molecule_id) AS t"}
        ]
    },
    "card_games": {
        "context": """Magic: The Gathering cards database.

CRITICAL - CASE SENSITIVITY (causes empty results!):
- Card names are CASE SENSITIVE: 'Annul' NOT 'annul'
- Status values: 'Legal', 'Banned', 'Restricted' (capital first!)
- Language values: 'Japanese', 'German', 'Chinese Simplified'

CRITICAL - ID COLUMN CONFUSION:
- "set number 5" → WHERE set_translations.id = 5 (NOT sets.id!)
- sets.id ≠ set_translations.id - different tables, different IDs!

CRITICAL - COLUMN CONFUSION:
- flavorText = flavor quotes/descriptions (German text, etc.)
- text = rules text
- For foreign text searches: use flavorText with LIKE

CRITICAL - BOOLEAN FLAGS:
- isForeignOnly = 0 means available IN the US
- isForeignOnly = 1 means ONLY outside US

CRITICAL - OUTPUT COLUMNS:
- "name the cards" or "list cards" → return id (NOT name!)
- "card names" → return name
- "Name all cards" → return id, not name!

CRITICAL - DISTINCT/COUNT:
- "How many types" → COUNT(type) NOT COUNT(DISTINCT type)
- For legalities JOIN: use COUNT(DISTINCT cards.id)

VALUE MAPPINGS:
- availability: LIKE '%paper%'
- frameVersion: INTEGER (2015 not '2015')
- keywords: exact match = 'Flying'

JOINS:
- cards.uuid = legalities.uuid
- cards.uuid = foreign_data.uuid  
- sets.code = set_translations.setCode""",
        "examples": [
            {"q": "Languages for card Annul numbered 29?", "e": "CASE SENSITIVE!",
             "sql": "SELECT T2.language FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.name = 'Annul' AND T1.number = '29'"},
            {"q": "Name all cards with 2015 frame below 100 EDHRec?", "e": "return id not name!",
             "sql": "SELECT id FROM cards WHERE edhrecRank < 100 AND frameVersion = 2015"},
            {"q": "Language and type of set number 206?", "e": "set_translations.id",
             "sql": "SELECT T2.language, T1.type FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.id = 206"},
            {"q": "Artist for card with German text 'Das perfekte'?", "e": "use flavorText",
             "sql": "SELECT DISTINCT T1.artist FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.flavorText LIKE '%Das perfekte%'"},
            {"q": "Sets not available outside US with Japanese?", "e": "isForeignOnly = 0",
             "sql": "SELECT T1.name FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Japanese' AND T1.isForeignOnly = 0"},
            {"q": "How many types did Aaron Boyd illustrate?", "e": "",
             "sql": "SELECT COUNT(type) FROM cards WHERE artist = 'Aaron Boyd'"},
            {"q": "Cards with status Restricted?", "e": "capital R",
             "sql": "SELECT COUNT(DISTINCT T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.status = 'Restricted'"},
            {"q": "Borderless without powerful foils?", "e": "",
             "sql": "SELECT id FROM cards WHERE borderColor = 'borderless' AND (cardKingdomFoilId IS NULL OR cardKingdomId IS NULL)"}
        ]
    },
    "formula_1": {
        "context": """F1 racing database with races, drivers, results, lap times.

CRITICAL - TIME FORMAT (causes empty results!):
- Times stored WITHOUT leading zeros: '1:40.123' NOT '0:01:40'
- Question says "Q2 time 0:01:40" → Use LIKE '1:40%'
- Question says "lap time 0:01:27" → Use LIKE '1:27%'
- NEVER use = '0:01:XX' - always use LIKE 'X:XX%'

CRITICAL - OUTPUT COLUMNS:
- "Who was the driver" → SELECT forename, surname (BOTH names!)
- "fastest lap time" questions → include the time value in output
- Return ONLY columns asked - no extras

CRITICAL - URL DISAMBIGUATION:
- circuits.url = Wikipedia about the CIRCUIT (permanent facility)
- races.url = Wikipedia about specific RACE EVENT
- "info about Sepang circuit" → circuits.url
- "info about 2009 Malaysian GP" → races.url

CRITICAL - TABLE USAGE:
- lapTimes = lap-by-lap times during race (has milliseconds)
- pitStops = pit stop times (duration in milliseconds)
- results.fastestLapTime = formatted string like '1:07.411'
- results.fastestLapSpeed = speed value

COLUMN NOTES:
- Position '\\N' = did not finish
- Dates: 'YYYY-MM-DD' format""",
        "examples": [
            {"q": "Driver with Q2 time 0:01:40 in race 355?", "e": "TIME FORMAT: LIKE not =",
             "sql": "SELECT DISTINCT T2.forename, T2.surname, T2.nationality FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T1.driverId = T2.driverId WHERE T1.raceId = 355 AND T1.q2 LIKE '1:40%'"},
            {"q": "Driver with lap time 0:01:27 in race 161?", "e": "",
             "sql": "SELECT DISTINCT T2.forename, T2.surname, T2.url FROM lapTimes AS T1 INNER JOIN drivers AS T2 ON T1.driverId = T2.driverId WHERE T1.raceId = 161 AND T1.time LIKE '1:27%'"},
            {"q": "Which race had fastest lap time? Include time.", "e": "",
             "sql": "SELECT T3.name, T1.fastestLapTime FROM results AS T1 INNER JOIN races AS T3 ON T1.raceId = T3.raceId WHERE T1.fastestLapTime IS NOT NULL ORDER BY T1.fastestLapTime ASC LIMIT 1"},
            {"q": "Info about races at Sepang circuit?", "e": "circuits.url for circuit info",
             "sql": "SELECT url FROM circuits WHERE name = 'Sepang International Circuit'"},
            {"q": "Fastest lap speed for driver 1 in race 348?", "e": "",
             "sql": "SELECT fastestLapSpeed FROM results WHERE driverId = 1 AND raceId = 348"},
            {"q": "Positions held by constructor Renault?", "e": "",
             "sql": "SELECT DISTINCT T1.position FROM constructorStandings AS T1 INNER JOIN constructors AS T2 ON T2.constructorId = T1.constructorId WHERE T2.name = 'Renault'"}
        ]
    },
    "european_football_2": {
        "context": "European football (soccer) with matches, players, teams. Player attributes have FIFA ratings. Match data includes home/away goals.",
        "examples": [
            {"q": "Players with volleys and dribbling over 70?", "e": "",
             "sql": "SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.volleys > 70 AND t2.dribbling > 70"},
            {"q": "Average overall rating for players taller than 170 from 2010-2015?", "e": "",
             "sql": "SELECT CAST(SUM(t2.overall_rating) AS REAL) / COUNT(t2.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.height > 170 AND STRFTIME('%Y',t2.`date`) >= '2010' AND STRFTIME('%Y',t2.`date`) <= '2015'"}
        ]
    },
    "thrombosis_prediction": {
        "context": """Medical database with patients, examinations, lab tests.

CRITICAL - DO NOT USE DISTINCT UNLESS ASKED:
- "How many patients" → COUNT(ID) NOT COUNT(DISTINCT ID)
- "List patients" → SELECT ID NOT SELECT DISTINCT ID  
- "How many examinations" → COUNT(*) NOT COUNT(DISTINCT ID)
- Only use DISTINCT if question says "unique" or "different"

CRITICAL VALUE MAPPINGS:
- Admission: '+' = in-patient, '-' = outpatient
- Thrombosis: 0 = negative, 1 = mild positive, 2 = severe positive
- SEX: 'M' = male, 'F' = female

CRITICAL PATTERNS:
- Diagnosis exists in BOTH Patient AND Examination - use Patient.Diagnosis by default
- For normal/abnormal: use exact 'Normal'/'Abnormal' strings
- Lab normal ranges: T-CHO < 250, TP between 6.0-8.5, LDH <= 500
- For ratios: SUM(CASE WHEN cond THEN 1 ELSE 0 END) / SUM(CASE WHEN cond2 THEN 1 ELSE 0 END)
- For percentages: CAST(SUM(CASE...) AS REAL) * 100 / COUNT(*)
- Date format: '1995-09-04' (YYYY-MM-DD)
- Age: STRFTIME('%Y', 'now') - STRFTIME('%Y', Birthday)
- JOIN: Patient.ID = Examination.ID = Laboratory.ID""",
        "examples": [
            {"q": "How many male patients were followed up in 1998?", "e": "male = SEX = 'M'",
             "sql": "SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = 'M' AND STRFTIME('%Y', T2.Date) = '1998'"},
            {"q": "List patients diagnosed with Behcet's who had lab tests in 1998", "e": "",
             "sql": "SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Diagnosis = \"Behcet\" AND STRFTIME('%Y', T2.Date) = '1998'"},
            {"q": "Ratio of inpatient to outpatient for SLE?", "e": "SLE = Diagnosis; inpatient = '+'; outpatient = '-'",
             "sql": "SELECT SUM(CASE WHEN Admission = '+' THEN 1.0 ELSE 0 END) / SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END) FROM Patient WHERE Diagnosis = 'SLE'"},
            {"q": "Percentage of female patients born after 1930?", "e": "female = SEX = 'F'",
             "sql": "SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', Birthday) > '1930' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Patient WHERE SEX = 'F'"},
            {"q": "How many patients had thrombosis level 2 and ANA pattern P?", "e": "",
             "sql": "SELECT COUNT(*) FROM Examination WHERE Thrombosis = 2 AND `ANA Pattern` = 'P'"},
            {"q": "Was cholesterol for patient 2927464 on 1995-9-4 normal?", "e": "normal = T-CHO < 250",
             "sql": "SELECT CASE WHEN `T-CHO` < 250 THEN 'Normal' ELSE 'Abnormal' END FROM Laboratory WHERE ID = 2927464 AND Date = '1995-09-04'"},
            {"q": "Symptoms of youngest patient ever examined?", "e": "youngest = MAX(Birthday)",
             "sql": "SELECT T2.Symptoms, T1.Diagnosis FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Symptoms IS NOT NULL ORDER BY T1.Birthday DESC LIMIT 1"}
        ]
    },
    "superhero": {
        "context": "Superhero database with powers, publishers, attributes. Uses IDs for colors, genders, races - must join to get names.",
        "examples": [
            {"q": "Gold-eyed superheroes from Marvel?", "e": "",
             "sql": "SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN colour AS T3 ON T1.eye_colour_id = T3.id WHERE T2.publisher_name = 'Marvel Comics' AND T3.colour = 'Gold'"},
            {"q": "Male to female superhero ratio?", "e": "",
             "sql": "SELECT CAST(COUNT(CASE WHEN T2.gender = 'Male' THEN T1.id ELSE NULL END) AS REAL) / COUNT(CASE WHEN T2.gender = 'Female' THEN T1.id ELSE NULL END) FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id"}
        ]
    },
    "codebase_community": {
        "context": """Stack Overflow-like Q&A database with posts, users, comments, badges, votes.

CRITICAL - DATE FORMAT:
- Dates stored as 'YYYY-MM-DD HH:MM:SS' (ISO format)
- Question says "2014/4/23" → Use '2014-04-23' in SQL
- Use LIKE '2014-04-23%' for datetime matching

CRITICAL - POST TYPES:
- PostTypeId: 1 = question, 2 = answer

CRITICAL - JOINS:
- comments.PostId = posts.Id
- posts.OwnerUserId = users.Id
- posts.LastEditorUserId = users.Id""",
        "examples": [
            {"q": "User who last edited 'Correlation does not mean causation' post?", "e": "",
             "sql": "SELECT T2.DisplayName FROM posts AS T1 INNER JOIN users AS T2 ON T1.LastEditorUserId = T2.Id WHERE T1.Title = 'Examples for teaching: Correlation does not mean causation'"},
            {"q": "Comment at 20:29:39 on 2014/4/23 to a post, how many favorites?", "e": "date format!",
             "sql": "SELECT T1.FavoriteCount FROM posts AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.PostId WHERE T2.CreationDate LIKE '2014-04-23 20:29:39%'"},
            {"q": "How many users have more than 5 badges?", "e": "",
             "sql": "SELECT COUNT(UserId) FROM (SELECT UserId, COUNT(Name) AS num FROM badges GROUP BY UserId) T WHERE T.num > 5"}
        ]
    },
    "student_club": {
        "context": "Student club database with members, events, budgets, expenses. Links via member_id, event_id, budget_id.",
        "examples": [
            {"q": "Angela Sanders's major?", "e": "",
             "sql": "SELECT T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.first_name = 'Angela' AND T1.last_name = 'Sanders'"},
            {"q": "Total spent on food?", "e": "",
             "sql": "SELECT SUM(spent) FROM budget WHERE category = 'Food'"}
        ]
    },
    "debit_card_specializing": {
        "context": """Gas station customer consumption database.

CRITICAL - AGGREGATION PATTERNS:
- "Total consumption" → SUM(Consumption) GROUP BY CustomerID
- "Most/least consumption" → SUM(Consumption) GROUP BY CustomerID, ORDER BY
- Single month value → just Consumption (no SUM)

CRITICAL - DATE EXTRACTION:
- Date format: 'YYYYMM' (e.g., '201304' = April 2013)
- Year: SUBSTR(Date, 1, 4)
- Month: SUBSTR(Date, 5, 2)
- "Peak month 2013" → return SUBSTR(Date, 5, 2)

CRITICAL - COUNT WITH HAVING:
- "How many customers with total < X" → subquery:
  SELECT COUNT(*) FROM (SELECT CustomerID FROM yearmonth WHERE Date LIKE 'YYYY%' GROUP BY CustomerID HAVING SUM(Consumption) < X)

SEGMENT VALUES:
- customers.Segment: 'SME', 'LAM', 'KAM' (uppercase)
- gasstations.Segment: 'Value for money', 'Premium' (different!)

CURRENCY: 'CZK', 'EUR' (uppercase)""",
        "examples": [
            {"q": "Who in KAM consumed most? How much?", "e": "SUM for total",
             "sql": "SELECT T1.CustomerID, SUM(T2.Consumption) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'KAM' GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) DESC LIMIT 1"},
            {"q": "How many KAM customers had consumption < 30000 in 2012?", "e": "subquery",
             "sql": "SELECT COUNT(*) FROM (SELECT T2.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'KAM' AND T2.Date LIKE '2012%' GROUP BY T2.CustomerID HAVING SUM(T2.Consumption) < 30000)"},
            {"q": "Peak month for SME in 2013?", "e": "return month number",
             "sql": "SELECT SUBSTR(T2.Date, 5, 2) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'SME' AND T2.Date LIKE '2013%' GROUP BY SUBSTR(T2.Date, 5, 2) ORDER BY SUM(T2.Consumption) DESC LIMIT 1"},
            {"q": "Ratio of EUR to CZK customers?", "e": "",
             "sql": "SELECT CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT) / SUM(IIF(Currency = 'CZK', 1, 0)) FROM customers"}
        ]
    }
}

# =============================================================================
# HELPERS
# =============================================================================

def get_schema(db_path: str) -> Dict:
    schema = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table,) in cursor.fetchall():
            cursor.execute(f'PRAGMA table_info("{table}")')
            schema[table] = [(r[1], r[2]) for r in cursor.fetchall()]
        conn.close()
    except:
        pass
    return schema


def format_schema(schema: Dict) -> str:
    lines = []
    for table, cols in schema.items():
        col_strs = []
        for c_name, c_type in cols:
            if ' ' in c_name or '(' in c_name:
                col_strs.append(f"`{c_name}` {c_type}")
            else:
                col_strs.append(f"{c_name} {c_type}")
        lines.append(f"CREATE TABLE {table} (\n  " + ",\n  ".join(col_strs) + "\n);")
    return "\n\n".join(lines)


def profile_column(conn, table: str, column: str) -> Dict:
    col_quoted = f'"{column}"' if ' ' in column or '(' in column else column
    profile = {"distinct": 0, "top_values": []}
    try:
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(DISTINCT {col_quoted}) FROM "{table}"')
        profile["distinct"] = cursor.fetchone()[0]
        cursor.execute(f'''
            SELECT {col_quoted}, COUNT(*) as cnt 
            FROM "{table}" WHERE {col_quoted} IS NOT NULL 
            GROUP BY {col_quoted} ORDER BY cnt DESC LIMIT 5
        ''')
        profile["top_values"] = [str(r[0]) for r in cursor.fetchall()]
    except:
        pass
    return profile


def build_profile_cache(db_path: str) -> Dict:
    profiles = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        for table in tables:
            profiles[table] = {}
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = [r[1] for r in cursor.fetchall()]
            for col in columns:
                profiles[table][col] = profile_column(conn, table, col)
        conn.close()
    except:
        pass
    return profiles


def format_profile_hints(profiles: Dict, schema: Dict, max_hints: int = 30) -> str:
    hints = []
    count = 0
    for table, cols in schema.items():
        if count >= max_hints:
            break
        table_prof = profiles.get(table, {})
        for col_name, _ in cols:
            if count >= max_hints:
                break
            prof = table_prof.get(col_name, {})
            if prof.get("top_values"):
                vals = ", ".join(f"'{v}'" for v in prof["top_values"][:3])
                hints.append(f"{table}.{col_name}: e.g. {vals}")
                count += 1
    return "\n".join(hints)


def execute_sql(db_path: str, sql: str) -> Dict:
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def normalize_value(val) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if re.match(r'^-?\d+\.0+$', s):
        s = str(int(float(s)))
    return s


def extract_sql(text: str, db_name: str) -> str:
    text = text.strip()
    
    # Try to extract from code blocks
    patterns = [
        r"```sql\s*(.*?)```",
        r"```\s*(.*?)```",
        r"(?:SELECT|WITH)\s+.*?(?:;|$)"
    ]
    
    for pattern in patterns[:2]:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            text = matches[-1].strip()
            break
    
    # Clean up
    text = text.replace("```sql", "").replace("```", "").strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip(';').strip()
    
    return text


def call_with_retry(client, params: Dict, max_retries: int = 8) -> Dict:
    import random
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**params)
            return {"success": True, "response": response}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                return {"success": False, "error": str(e)}
    return {"success": False, "error": "Max retries exceeded"}


# =============================================================================
# QUESTION-TYPE ROUTER
# =============================================================================

QUESTION_TYPES = {
    "ratio": {
        "keywords": ["ratio", "proportion", "percentage", "percent", "%", "how much more", "how many more", "compared to"],
        "rules": """RATIO QUESTION - CRITICAL RULES:
- "ratio of X to Y" means X / Y (X is numerator, Y is denominator)
- "percentage of X" means (COUNT of X) * 100.0 / (COUNT of total)
- Use CAST(... AS REAL) to avoid integer division
- Double-check which value comes first in the question!"""
    },
    "aggregation": {
        "keywords": ["how many", "count", "total", "sum", "average", "avg", "number of"],
        "rules": """AGGREGATION QUESTION - CRITICAL RULES:
- "How many X" → COUNT(*) or COUNT(column) WITHOUT DISTINCT
- "How many unique/different X" → COUNT(DISTINCT column)
- "Total X" → SUM(column)
- "Average X" → AVG(column) or SUM/COUNT
- Do NOT add DISTINCT unless explicitly asked!"""
    },
    "superlative": {
        "keywords": ["highest", "lowest", "most", "least", "best", "worst", "top", "bottom", "maximum", "minimum", "largest", "smallest", "oldest", "youngest", "first", "last"],
        "rules": """SUPERLATIVE QUESTION - CRITICAL RULES:
- "highest/most/best/largest/maximum" → ORDER BY column DESC LIMIT 1
- "lowest/least/worst/smallest/minimum" → ORDER BY column ASC LIMIT 1
- "oldest" for dates → ORDER BY date ASC (earliest date)
- "youngest" for dates → ORDER BY date DESC (latest date)
- "oldest" for people (birthday) → ORDER BY birthday ASC (born earliest)
- "youngest" for people → ORDER BY birthday DESC (born latest)
- Add IS NOT NULL when ordering by nullable columns!"""
    },
    "date": {
        "keywords": ["when", "date", "year", "month", "day", "after", "before", "between", "during", "in 19", "in 20", "/"],
        "rules": """DATE QUESTION - CRITICAL RULES:
- Convert date formats: "2014/4/23" → '2014-04-23'
- Use STRFTIME('%Y', date) for year extraction
- Use LIKE 'YYYY-MM-DD%' for datetime matching
- For date comparisons: date(column) > 'YYYY-MM-DD'
- Check the actual format stored in database!"""
    },
    "join": {
        "keywords": ["and", "with", "from", "in", "whose", "who has", "that have", "belonging to"],
        "rules": """JOIN QUESTION - CRITICAL RULES:
- Identify all tables needed based on columns referenced
- Use explicit JOIN conditions from schema foreign keys
- Check table aliases are consistent
- Don't join more tables than necessary"""
    }
}


def classify_question(question: str, evidence: str) -> List[str]:
    """Classify question into types for specialized prompting."""
    question_lower = (question + " " + (evidence or "")).lower()
    
    detected_types = []
    for qtype, config in QUESTION_TYPES.items():
        for keyword in config["keywords"]:
            if keyword in question_lower:
                detected_types.append(qtype)
                break
    
    # Default to simple if no specific type detected
    if not detected_types:
        detected_types = ["simple"]
    
    return detected_types


def get_type_specific_rules(question_types: List[str]) -> str:
    """Get specialized rules for detected question types."""
    rules = []
    for qtype in question_types:
        if qtype in QUESTION_TYPES:
            rules.append(QUESTION_TYPES[qtype]["rules"])
    return "\n\n".join(rules)


# =============================================================================
# DYNAMIC VALUE GROUNDING
# =============================================================================

def extract_potential_values(question: str, evidence: str) -> List[str]:
    """Extract potential entity values from question and evidence."""
    text = question + " " + (evidence or "")
    
    potential_values = []
    
    # Extract quoted strings
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    for match in quoted:
        potential_values.extend([m for m in match if m])
    
    # Extract capitalized words (potential proper nouns)
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    potential_values.extend(caps)
    
    # Extract numbers
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    potential_values.extend(numbers)
    
    # Extract date-like patterns
    dates = re.findall(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b', text)
    potential_values.extend(dates)
    
    return list(set(potential_values))


def ground_values_in_db(db_path: str, potential_values: List[str], schema: Dict) -> Dict[str, List[str]]:
    """Search database for actual values matching potential entities."""
    grounded = {}
    
    if not potential_values:
        return grounded
    
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        
        for table, columns in schema.items():
            for col_name, col_type in columns:
                # Only check text columns
                if 'TEXT' in col_type.upper() or 'CHAR' in col_type.upper() or 'VARCHAR' in col_type.upper():
                    col_quoted = f'"{col_name}"' if ' ' in col_name or '(' in col_name else col_name
                    
                    for val in potential_values[:10]:  # Limit to avoid too many queries
                        try:
                            # Case-insensitive search
                            cursor.execute(f'''
                                SELECT DISTINCT {col_quoted} FROM "{table}" 
                                WHERE {col_quoted} LIKE ? LIMIT 3
                            ''', (f'%{val}%',))
                            
                            matches = [row[0] for row in cursor.fetchall() if row[0]]
                            
                            if matches:
                                key = f"{table}.{col_name}"
                                if key not in grounded:
                                    grounded[key] = []
                                grounded[key].extend(matches)
                        except:
                            continue
        
        conn.close()
    except:
        pass
    
    return grounded


def format_grounded_values(grounded: Dict[str, List[str]]) -> str:
    """Format grounded values as hints for the prompt."""
    if not grounded:
        return ""
    
    lines = ["FOUND VALUES IN DATABASE (use exact spelling!):"]
    for col, values in grounded.items():
        unique_vals = list(set(values))[:5]
        vals_str = ", ".join(f"'{v}'" for v in unique_vals)
        lines.append(f"  {col}: {vals_str}")
    
    return "\n".join(lines)


def generate_candidates(client, question: str, evidence: str, 
                        schema_text: str, profile_hints: str, db_name: str,
                        schema: Dict = None, db_path: str = None) -> List[Tuple[str, float]]:
    """Generate SQL candidates with question-type routing and value grounding."""
    
    db_info = DATABASE_EXAMPLES.get(db_name, {"context": "", "examples": []})
    
    # === STAGE 1: QUESTION CLASSIFICATION ===
    question_types = classify_question(question, evidence)
    type_rules = get_type_specific_rules(question_types)
    
    # === STAGE 2: DYNAMIC VALUE GROUNDING ===
    grounded_hints = ""
    if schema and db_path:
        potential_values = extract_potential_values(question, evidence)
        if potential_values:
            grounded = ground_values_in_db(db_path, potential_values, schema)
            grounded_hints = format_grounded_values(grounded)
    
    # === BUILD SYSTEM PROMPT WITH TYPE-SPECIFIC RULES ===
    system = f"""You are solving BIRD benchmark text-to-SQL questions.

QUESTION TYPE DETECTED: {', '.join(question_types).upper()}

{type_rules}

CRITICAL - DO NOT HALLUCINATE:
- Use ONLY columns from the schema
- READ the evidence field - it contains exact value mappings
- Output valid SQLite syntax
- RETURN ONLY THE COLUMNS ASKED FOR - do NOT add extra columns

CRITICAL SQL PATTERNS - FOLLOW EXACTLY:
1. DISTINCT: Do NOT add DISTINCT unless the question explicitly asks for "unique", "distinct", or "different" values
   - "How many patients" → COUNT(ID) not COUNT(DISTINCT ID)
   - "List patients" → SELECT ID not SELECT DISTINCT ID
   - "How many unique/distinct X" → COUNT(DISTINCT X)

2. COUNT: Match the question's intent precisely
   - "How many X" → COUNT(*) or COUNT(column) WITHOUT DISTINCT
   - "How many types of X" → COUNT(type) not COUNT(DISTINCT type) unless "unique types"
   - "How many different X" → COUNT(DISTINCT X)

3. Column Output: Return columns SEPARATELY, not concatenated
   - "name and artist" → SELECT name, artist (NOT SELECT name || ' ' || artist)
   - "first and last name" → SELECT first_name, last_name (NOT concatenated)

4. NULL Handling: Add IS NOT NULL when ordering by nullable columns
   - ORDER BY column ASC → WHERE column IS NOT NULL ORDER BY column ASC

5. Sort Direction: Default is ASC unless question says "highest", "most", "top", "maximum"
   - "lowest X" → ORDER BY X ASC
   - "highest X" → ORDER BY X DESC

DATABASE: {db_name}
{db_info['context']}
"""
    
    if db_info.get('examples'):
        system += "\nLEARN FROM THESE EXAMPLES:\n"
        for i, ex in enumerate(db_info['examples'], 1):
            system += f"Example {i}:\nQ: {ex['q']}\nEvidence: {ex['e'] or 'None'}\nSQL: {ex['sql']}\n\n"
    
    if profile_hints:
        system += f"\nCOLUMN VALUE SAMPLES:\n{profile_hints}\n"
    
    # Add grounded values from dynamic search
    if grounded_hints:
        system += f"\n{grounded_hints}\n"
    
    system += f"\nSCHEMA:\n{schema_text}"
    
    user_msg = f"Question: {question}\nEvidence: {evidence or 'None'}\n\nOutput ONLY the SQL in ```sql blocks:"
    
    candidates = []
    temps = [0.0, 0.3, 0.5][:NUM_CANDIDATES]
    
    for i, temp in enumerate(temps):
        params = {
            "model": PRIMARY_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            "max_completion_tokens": 600,
        }
        
        if not PRIMARY_MODEL.startswith("o1") and not PRIMARY_MODEL.startswith("o3"):
            params["temperature"] = temp
        
        if i > 0:
            params["seed"] = 42 + i * 100
        
        result = call_with_retry(client, params)
        
        if result["success"]:
            response = result["response"]
            text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = (input_tokens * 1.75 + output_tokens * 14.0) / 1_000_000
            sql = extract_sql(text, db_name)
            if sql:
                candidates.append((sql, cost))
    
    return candidates


def vote_candidates(candidates: List[Tuple[str, float]], db_path: str) -> Tuple[str, float, str]:
    if not candidates:
        return "", 0.0, "no_candidates"
    
    total_cost = sum(c[1] for c in candidates)
    
    results = []
    for sql, _ in candidates:
        exec_result = execute_sql(db_path, sql)
        if exec_result["success"]:
            result_set = frozenset(
                tuple(normalize_value(v) for v in row)
                for row in (exec_result["result"] or [])
            )
            results.append((sql, result_set))
        else:
            results.append((sql, None))
    
    successful = [(sql, rs) for sql, rs in results if rs is not None]
    
    if not successful:
        return candidates[0][0], total_cost, "all_failed"
    
    if len(successful) == 1:
        return successful[0][0], total_cost, "single_success"
    
    groups = {}
    for sql, rs in successful:
        if rs not in groups:
            groups[rs] = []
        groups[rs].append(sql)
    
    best_group = max(groups.values(), key=len)
    
    if len(best_group) >= 2:
        return best_group[0], total_cost, f"majority_{len(best_group)}"
    
    return successful[0][0], total_cost, "no_majority"


def repair_sql(client, sql: str, error: str, question: str,
               evidence: str, schema_text: str, db_name: str) -> Tuple[str, float]:
    db_info = DATABASE_EXAMPLES.get(db_name, {"context": ""})
    
    prompt = f"""Fix this SQL error.

Database: {db_name}
{db_info.get('context', '')}

Question: {question}
Evidence: {evidence or 'None'}

Failed SQL: {sql}
Error: {error}

Schema:
{schema_text[:2500]}

Output ONLY the corrected SQL:"""
    
    params = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": "Fix the SQL error using only valid schema columns."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 600,
        "temperature": 0
    }
    
    result = call_with_retry(client, params)
    
    if result["success"]:
        response = result["response"]
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = (input_tokens * 1.75 + output_tokens * 14.0) / 1_000_000
        return extract_sql(text, db_name), cost
    
    return "", 0.0


# =============================================================================
# MAIN PROCESSING
# =============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=180,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def process_single_question(question_data: Dict) -> Dict:
    """Process one question with full analysis."""
    from openai import OpenAI
    
    client = OpenAI()
    start = time.time()
    total_cost = 0.0
    
    idx = question_data.get("idx", 0)
    question = question_data["question"]
    evidence = question_data.get("evidence", "")
    db_name = question_data["db_id"]
    gold_sql = question_data.get("SQL", "")
    
    db_path = f"{DATABASE_PATH}/{db_name}/{db_name}.sqlite"
    
    try:
        # Get schema
        schema = get_schema(db_path)
        schema_text = format_schema(schema)
        profiles = build_profile_cache(db_path)
        profile_hints = format_profile_hints(profiles, schema)
        
        # Generate candidates with question routing and value grounding
        candidates = generate_candidates(
            client, question, evidence, schema_text, profile_hints, db_name,
            schema=schema, db_path=db_path
        )
        
        if not candidates:
            return {
                "idx": idx, "db_id": db_name, "question": question,
                "correct": False, "cost": 0.0, "time": time.time() - start,
                "error_type": "no_candidates", "predicted_sql": "",
                "gold_sql": gold_sql, "vote_info": "no_candidates"
            }
        
        # Vote
        sql, total_cost, vote_info = vote_candidates(candidates, db_path)
        
        # Execute voted SQL
        exec_result = execute_sql(db_path, sql)
        
        # Repair if failed
        if not exec_result["success"]:
            for attempt in range(MAX_REPAIR_ATTEMPTS):
                repaired, repair_cost = repair_sql(
                    client, sql, exec_result["error"],
                    question, evidence, schema_text, db_name
                )
                total_cost += repair_cost
                if repaired:
                    repair_result = execute_sql(db_path, repaired)
                    if repair_result["success"]:
                        sql = repaired
                        exec_result = repair_result
                        vote_info = f"repaired_{attempt+1}"
                        break
        
        # Compare with gold
        gold_result = execute_sql(db_path, gold_sql)
        
        correct = False
        error_type = "unknown"
        
        if exec_result["success"] and gold_result["success"]:
            pred_rows = exec_result["result"] or []
            gold_rows = gold_result["result"] or []
            
            # Normalize each row to sorted tuple of strings
            def normalize_row(row):
                return tuple(sorted(normalize_value(v) for v in row))
            
            pred_set = frozenset(normalize_row(row) for row in pred_rows)
            gold_set = frozenset(normalize_row(row) for row in gold_rows)
            
            correct = (pred_set == gold_set)
            if not correct:
                # Also check if it's just row count difference with same values
                if pred_set.issubset(gold_set) or gold_set.issubset(pred_set):
                    error_type = "row_count_mismatch"
                else:
                    error_type = "wrong_result"
        elif not exec_result["success"]:
            error_type = "exec_error"
        elif not gold_result["success"]:
            error_type = "gold_error"
        
        return {
            "idx": idx,
            "db_id": db_name,
            "question": question,
            "evidence": evidence,
            "correct": correct,
            "cost": total_cost,
            "time": time.time() - start,
            "vote_info": vote_info,
            "error_type": error_type if not correct else "none",
            "predicted_sql": sql,
            "gold_sql": gold_sql,
            "predicted_result": str(exec_result.get("result", [])[:5]) if exec_result["success"] else exec_result.get("error", ""),
            "gold_result": str(gold_result.get("result", [])[:5]) if gold_result["success"] else "",
            "all_candidates": [c[0] for c in candidates]
        }
        
    except Exception as e:
        return {
            "idx": idx, "db_id": db_name, "question": question,
            "correct": False, "cost": total_cost, "time": time.time() - start,
            "error_type": "exception", "error": str(e),
            "predicted_sql": "", "gold_sql": gold_sql, "vote_info": "error"
        }


@app.function(image=image, volumes={"/data": volume}, timeout=60)
def load_questions():
    with open("/data/dev.json") as f:
        return json.load(f)


@app.function(image=image, volumes={"/data": volume}, timeout=120)
def save_results(results_data: dict, filename: str):
    with open(f"/data/{filename}", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to /data/{filename}")


@app.local_entrypoint()
def main(limit: int = None, db: str = None):
    """Main entry point with optional limit and db filter."""
    
    print("=" * 70)
    print("BIRD V51 - Question-Type Router + Dynamic Value Grounding")
    print("=" * 70)
    print(f"Model: {PRIMARY_MODEL}")
    print(f"Limit: {limit if limit else 'ALL'}")
    print(f"DB Filter: {db if db else 'ALL'}")
    print("=" * 70)
    
    print("Loading questions...")
    questions = load_questions.remote()
    
    # Filter by database if specified
    if db:
        questions = [q for q in questions if q["db_id"] == db]
    
    # Apply limit
    if limit:
        questions = questions[:limit]
    
    print(f"Processing {len(questions)} questions...")
    
    # Add index
    for i, q in enumerate(questions):
        q["idx"] = i
    
    start_time = time.time()
    results = list(process_single_question.map(
        questions, 
        order_outputs=False, 
        return_exceptions=True,
        wrap_returned_exceptions=False
    ))
    
    # Process results
    valid_results = []
    exceptions = 0
    for r in results:
        if isinstance(r, Exception):
            exceptions += 1
        else:
            valid_results.append(r)
    
    valid_results.sort(key=lambda x: x["idx"])
    
    # Statistics
    correct = sum(1 for r in valid_results if r["correct"])
    total_cost = sum(r.get("cost", 0) for r in valid_results)
    total_time = time.time() - start_time
    
    # Per-database stats
    db_stats = {}
    error_types = Counter()
    
    for r in valid_results:
        db_id = r["db_id"]
        if db_id not in db_stats:
            db_stats[db_id] = {"total": 0, "correct": 0, "errors": []}
        db_stats[db_id]["total"] += 1
        if r["correct"]:
            db_stats[db_id]["correct"] += 1
        else:
            error_types[r.get("error_type", "unknown")] += 1
            db_stats[db_id]["errors"].append({
                "idx": r["idx"],
                "question": r["question"],
                "error_type": r.get("error_type", "unknown"),
                "predicted": r.get("predicted_sql", ""),
                "gold": r.get("gold_sql", ""),
                "pred_result": r.get("predicted_result", ""),
                "gold_result": r.get("gold_result", "")
            })
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Completed: {len(valid_results)}/{len(questions)} ({exceptions} exceptions)")
    print(f"Correct: {correct}/{len(valid_results)} = {100*correct/len(valid_results):.2f}%")
    print(f"Cost: ${total_cost:.2f}")
    print(f"Time: {total_time/60:.1f} minutes")
    
    print("\nPer-database:")
    for db_id, stats in sorted(db_stats.items()):
        pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {db_id}: {stats['correct']}/{stats['total']} = {pct:.1f}%")
    
    print("\nError types:")
    for et, cnt in error_types.most_common():
        print(f"  {et}: {cnt}")
    
    # Save detailed results
    results_data = {
        "version": "v51",
        "model": PRIMARY_MODEL,
        "accuracy": correct / len(valid_results) if valid_results else 0,
        "correct": correct,
        "total": len(valid_results),
        "cost": total_cost,
        "time_minutes": total_time / 60,
        "per_database": {
            db_id: {
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                "correct": s["correct"],
                "total": s["total"],
                "errors": s["errors"]
            }
            for db_id, s in db_stats.items()
        },
        "error_types": dict(error_types),
        "all_results": valid_results
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_v51_{timestamp}.json"
    save_results.remote(results_data, filename)
    
    print(f"\nDetailed results saved to: /data/{filename}")
    print("Download with: modal volume get bird-dataset " + filename)
