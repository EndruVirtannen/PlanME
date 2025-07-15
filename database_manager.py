# database_manager.py - Opravená verzia s kompatibilnou MySQL syntaxou

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime, date, time, timedelta
import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd

# Databázové nastavenia
DB_CONFIG = {
    'host': 'DB_NAME',
    'database': 'sql7789559',
    'user': 'DB_USER',
    'password': 'DB_PASSWORD',
    'port': 3306,
    'charset': 'utf8mb4'
}


def get_db_connection():
    """Vytvorí spojenie s databázou"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Chyba pri spojení s databázou: {e}")
        return None


def check_and_recreate_tables() -> bool:
    """Skontroluje a znovu vytvorí tabuľky ak majú nesprávnu štruktúru"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Kontrola štruktúry tabuľky employees
        try:
            cursor.execute("DESCRIBE employees")
            columns = [row[0] for row in cursor.fetchall()]

            required_columns = ['id', 'name', 'team_id', 'email', 'phone', 'hourly_rate']
            missing_columns = [col for col in required_columns if col not in columns]

            if missing_columns:
                print(f"🔄 Tabuľka employees má nesprávnu štruktúru, znovu vytváram...")
                cursor.execute("DROP TABLE IF EXISTS employees")

        except Error:
            print("🔄 Tabuľka employees neexistuje, vytváram...")

        # Podobne pre ostatné tabuľky
        for table_name in ['teams', 'shift_types', 'vacation_requests', 'schedules', 'collaborations',
                           'schedule_history']:
            try:
                cursor.execute(f"DESCRIBE {table_name}")
            except Error:
                print(f"🔄 Tabuľka {table_name} neexistuje alebo má problémy")
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        connection.commit()
        return True

    except Error as e:
        print(f"Chyba pri kontrole tabuliek: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def init_database() -> bool:
    """Inicializuje databázové tabuľky"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # 1. Tabuľka teams
        teams_table = """
        CREATE TABLE IF NOT EXISTS teams (
            id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            manager_id VARCHAR(50),
            priority INT DEFAULT 1,
            color VARCHAR(7) DEFAULT '#4CAF50',
            department VARCHAR(100),
            location VARCHAR(100),
            min_coverage TEXT,
            max_coverage TEXT,
            target_coverage TEXT,
            weekend_multiplier DECIMAL(3,2) DEFAULT 1.00,
            holiday_multiplier DECIMAL(3,2) DEFAULT 0.50,
            supervisor_required BOOLEAN DEFAULT FALSE,
            emergency_contact BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT NULL
        )
        """

        # 2. Tabuľka employees
        employees_table = """
        CREATE TABLE IF NOT EXISTS employees (
            id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            team_id VARCHAR(50),
            email VARCHAR(255),
            phone VARCHAR(50),
            hourly_rate DECIMAL(8,2) DEFAULT 15.00,
            max_consecutive_days INT DEFAULT 5,
            max_night_shifts INT DEFAULT 8,
            annual_vacation_days INT DEFAULT 25,
            contract_type VARCHAR(50) DEFAULT 'FULL_TIME',
            overtime_eligible BOOLEAN DEFAULT TRUE,
            weekend_work_allowed BOOLEAN DEFAULT TRUE,
            night_shift_restriction BOOLEAN DEFAULT FALSE,
            seniority_years INT DEFAULT 0,
            performance_rating DECIMAL(3,2) DEFAULT 3.00,
            monthly_hours_target INT DEFAULT 160,
            weekly_hours_min INT DEFAULT 20,
            weekly_hours_max INT DEFAULT 48,
            preferences TEXT,
            skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_team_id (team_id)
        )
        """

        # 3. Tabuľka shift_types
        shift_types_table = """
        CREATE TABLE IF NOT EXISTS shift_types (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            rest_days_after INT DEFAULT 0,
            required_skills TEXT,
            min_skill_level VARCHAR(50) DEFAULT 'BEGINNER',
            premium_pay DECIMAL(4,3) DEFAULT 0.000,
            is_weekend_applicable BOOLEAN DEFAULT TRUE,
            max_consecutive_days INT DEFAULT 5,
            min_employees INT DEFAULT 1,
            max_employees INT DEFAULT 3,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        # 4. Tabuľka vacation_requests
        vacation_requests_table = """
        CREATE TABLE IF NOT EXISTS vacation_requests (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(50) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            vacation_type VARCHAR(50) DEFAULT 'ANNUAL',
            reason TEXT,
            priority VARCHAR(20) DEFAULT 'MEDIUM',
            approved BOOLEAN DEFAULT FALSE,
            created_date DATE NOT NULL,
            approved_by VARCHAR(50),
            approved_date DATE,
            INDEX idx_employee_id (employee_id)
        )
        """

        # 5. Tabuľka schedules
        schedules_table = """
        CREATE TABLE IF NOT EXISTS schedules (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(50) NOT NULL,
            shift_date DATE NOT NULL,
            shift_type_id INT NOT NULL,
            team_id VARCHAR(50),
            hours_worked DECIMAL(4,2),
            cost DECIMAL(10,2),
            is_weekend BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_employee_date (employee_id, shift_date),
            INDEX idx_employee (employee_id),
            INDEX idx_shift_type (shift_type_id),
            INDEX idx_team (team_id)
        )
        """

        # 6. Tabuľka collaborations
        collaborations_table = """
        CREATE TABLE IF NOT EXISTS collaborations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            team1_id VARCHAR(50) NOT NULL,
            team2_id VARCHAR(50) NOT NULL,
            allowed_shift_types TEXT,
            max_shared_employees INT DEFAULT 2,
            priority INT DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_team1 (team1_id),
            INDEX idx_team2 (team2_id)
        )
        """

        # 7. Tabuľka schedule_history
        schedule_history_table = """
        CREATE TABLE IF NOT EXISTS schedule_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            schedule_name VARCHAR(255) NOT NULL,
            period_start DATE NOT NULL,
            period_end DATE NOT NULL,
            total_employees INT,
            total_hours DECIMAL(10,2),
            total_cost DECIMAL(12,2),
            schedule_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        # Vytvorenie tabuliek v správnom poradí (bez FK constraints kvôli kompatibilite)
        tables = [
            ("teams", teams_table),
            ("employees", employees_table),
            ("shift_types", shift_types_table),
            ("vacation_requests", vacation_requests_table),
            ("schedules", schedules_table),
            ("collaborations", collaborations_table),
            ("schedule_history", schedule_history_table)
        ]

        # Najprv vyčistíme všetky tabuľky kvôli možným zmenám štruktúry
        cleanup_tables = [
            "schedules", "vacation_requests", "collaborations",
            "employees", "shift_types", "teams", "schedule_history"
        ]

        for table_name in cleanup_tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"🗑️ Tabuľka {table_name} vymazaná")
            except Error as e:
                print(f"⚠️ Chyba pri mazaní tabuľky {table_name}: {e}")

        # Teraz vytvoríme nové tabuľky
        for table_name, table_sql in tables:
            try:
                cursor.execute(table_sql)
                print(f"✅ Tabuľka {table_name} vytvorená")
            except Error as e:
                print(f"❌ Chyba pri vytváraní tabuľky {table_name}: {e}")

        connection.commit()
        return True

    except Error as e:
        st.error(f"Chyba pri inicializácii databázy: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def save_teams_to_db(teams_data: List[Dict]) -> bool:
    """Uloží tímy do databázy"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Kontrola štruktúry tabuľky
        try:
            cursor.execute("DESCRIBE teams")
            columns = [row[0] for row in cursor.fetchall()]
            if 'name' not in columns:
                print("❌ Tabuľka teams nemá správnu štruktúru, reinicializujem...")
                init_database()
                cursor = connection.cursor()
        except Error:
            print("❌ Tabuľka teams neexistuje, reinicializujem...")
            init_database()
            cursor = connection.cursor()

        # Najprv vyčistíme existujúce dáta
        cursor.execute("DELETE FROM teams")

        # Vložíme nové dáta
        insert_query = """
        INSERT INTO teams (
            id, name, description, priority, color, department, location,
            min_coverage, max_coverage, target_coverage, weekend_multiplier,
            holiday_multiplier, supervisor_required, emergency_contact
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for team in teams_data:
            values = (
                team.get('id', ''),
                team.get('name', ''),
                team.get('description', ''),
                int(team.get('priority', 1)),
                team.get('color', '#4CAF50'),
                team.get('department', ''),
                team.get('location', ''),
                json.dumps(team.get('min_coverage', {}), ensure_ascii=False),
                json.dumps(team.get('max_coverage', {}), ensure_ascii=False),
                json.dumps(team.get('target_coverage', {}), ensure_ascii=False),
                float(team.get('weekend_multiplier', 1.0)),
                float(team.get('holiday_multiplier', 0.5)),
                bool(team.get('supervisor_required', False)),
                bool(team.get('emergency_contact', True))
            )

            try:
                cursor.execute(insert_query, values)
                print(f"✅ Tím {team.get('name', 'N/A')} uložený")
            except Error as team_error:
                print(f"❌ Chyba  pri vkladaní tímu {team.get('name', 'N/A')}: {team_error}")
                continue

        connection.commit()
        return True

    except Error as e:
        st.error(f"Chyba pri ukladaní tímov: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def save_employees_to_db(employees_data: List[Dict]) -> bool:
    """Uloží zamestnancov do databázy"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Najprv skontrolujeme či tabuľka existuje so správnou štruktúrou
        try:
            cursor.execute("SHOW COLUMNS FROM employees")
            columns = [row[0] for row in cursor.fetchall()]
            print(f"🔍 Stĺpce v tabuľke employees: {columns}")

            if 'name' not in columns:
                print("❌ Tabuľka employees nemá stĺpec 'name', znovu vytváram...")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                cursor.execute("DROP TABLE IF EXISTS employees")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

                # Vytvoríme tabuľku znovu
                employees_table = """
                CREATE TABLE employees (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    team_id VARCHAR(50),
                    email VARCHAR(255),
                    phone VARCHAR(50),
                    hourly_rate DECIMAL(8,2) DEFAULT 15.00,
                    max_consecutive_days INT DEFAULT 5,
                    max_night_shifts INT DEFAULT 8,
                    annual_vacation_days INT DEFAULT 25,
                    contract_type VARCHAR(50) DEFAULT 'FULL_TIME',
                    overtime_eligible BOOLEAN DEFAULT TRUE,
                    weekend_work_allowed BOOLEAN DEFAULT TRUE,
                    night_shift_restriction BOOLEAN DEFAULT FALSE,
                    seniority_years INT DEFAULT 0,
                    performance_rating DECIMAL(3,2) DEFAULT 3.00,
                    monthly_hours_target INT DEFAULT 160,
                    weekly_hours_min INT DEFAULT 20,
                    weekly_hours_max INT DEFAULT 48,
                    preferences TEXT,
                    skills TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_team_id (team_id)
                )
                """
                cursor.execute(employees_table)
                print("✅ Tabuľka employees znovu vytvorená so správnou štruktúrou")

        except Error as e:
            print(f"❌ Problém s tabuľkou employees: {e}")
            return False

        # Najprv vyčistíme existujúce dáta
        cursor.execute("DELETE FROM employees")

        # Vložíme nové dáta
        insert_query = """
        INSERT INTO employees (
            id, name, team_id, email, phone, hourly_rate, max_consecutive_days,
            max_night_shifts, annual_vacation_days, contract_type, overtime_eligible,
            weekend_work_allowed, night_shift_restriction, seniority_years,
            performance_rating, monthly_hours_target, weekly_hours_min,
            weekly_hours_max, preferences, skills
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for emp in employees_data:
            # Zabezpečíme všetky potrebné hodnoty s defaultmi
            values = (
                emp.get('id', ''),
                emp.get('name', ''),
                emp.get('team_id', ''),
                emp.get('email', ''),
                emp.get('phone', ''),
                float(emp.get('hourly_rate', 15.0)),
                int(emp.get('max_cons', 5)),
                int(emp.get('max_night', 8)),
                int(emp.get('annual_vacation', 25)),
                'FULL_TIME',  # Default contract type
                bool(emp.get('overtime_eligible', True)),
                bool(emp.get('weekend_allowed', True)),
                bool(emp.get('night_restriction', False)),
                int(emp.get('seniority', 0)),
                float(emp.get('performance', 3.0)),
                int(emp.get('monthly_target', 160)),
                int(emp.get('weekly_min', 20)),
                int(emp.get('weekly_max', 48)),
                json.dumps(emp.get('preferences', []), ensure_ascii=False),
                json.dumps(emp.get('skills', []), ensure_ascii=False)
            )

            try:
                cursor.execute(insert_query, values)
                print(f"✅ Zamestnanec {emp.get('name', 'N/A')} uložený")
            except Error as emp_error:
                print(f"❌ Chyba pri vkladaní zamestnanca {emp.get('name', 'N/A')}: {emp_error}")
                continue

        connection.commit()
        return True

    except Error as e:
        st.error(f"Chyba pri ukladaní zamestnancov: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def save_shifts_to_db(shifts_data: List[Dict]) -> bool:
    """Uloží smeny do databázy"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Najprv vyčistíme existujúce dáta
        cursor.execute("DELETE FROM shift_types")

        # Vložíme nové dáta
        insert_query = """
        INSERT INTO shift_types (
            name, start_time, end_time, rest_days_after, required_skills,
            min_skill_level, premium_pay, is_weekend_applicable,
            max_consecutive_days, min_employees, max_employees
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for shift in shifts_data:
            values = (
                shift.get('name', ''),
                shift.get('start'),
                shift.get('end'),
                int(shift.get('rest_days', 0)),
                json.dumps(shift.get('skills', []), ensure_ascii=False),
                shift.get('min_level', 'BEGINNER'),
                float(shift.get('premium', 0.0)),
                bool(shift.get('weekend_applicable', True)),
                int(shift.get('max_consecutive', 5)),
                int(shift.get('min_employees', 1)),
                int(shift.get('max_employees', 3))
            )

            try:
                cursor.execute(insert_query, values)
            except Error as shift_error:
                st.error(f"Chyba pri vkladaní smeny {shift.get('name', 'N/A')}: {shift_error}")
                continue

        connection.commit()
        return True

    except Error as e:
        st.error(f"Chyba pri ukladaní smien: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def load_teams_from_db() -> List[Dict]:
    """Načíta tímy z databázy"""
    connection = get_db_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM teams")
        teams = cursor.fetchall()

        # Konvertujeme JSON stringy späť na objekty
        for team in teams:
            if team.get('min_coverage'):
                try:
                    team['min_coverage'] = json.loads(team['min_coverage'])
                except:
                    team['min_coverage'] = {}
            if team.get('max_coverage'):
                try:
                    team['max_coverage'] = json.loads(team['max_coverage'])
                except:
                    team['max_coverage'] = {}
            if team.get('target_coverage'):
                try:
                    team['target_coverage'] = json.loads(team['target_coverage'])
                except:
                    team['target_coverage'] = {}

        return teams

    except Error as e:
        st.error(f"Chyba pri načítavaní tímov: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def load_employees_from_db() -> List[Dict]:
    """Načíta zamestnancov z databázy"""
    connection = get_db_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM employees")
        employees = cursor.fetchall()

        # Konvertujeme JSON stringy späť na objekty
        for emp in employees:
            if emp.get('preferences'):
                try:
                    emp['preferences'] = json.loads(emp['preferences'])
                except:
                    emp['preferences'] = []
            if emp.get('skills'):
                try:
                    emp['skills'] = json.loads(emp['skills'])
                except:
                    emp['skills'] = []

            # Mapovanie názvov stĺpcov
            emp['max_cons'] = emp.get('max_consecutive_days', 5)
            emp['max_night'] = emp.get('max_night_shifts', 8)
            emp['monthly_target'] = emp.get('monthly_hours_target', 160)
            emp['weekly_min'] = emp.get('weekly_hours_min', 20)
            emp['weekly_max'] = emp.get('weekly_hours_max', 48)
            emp['overtime_eligible'] = emp.get('overtime_eligible', True)
            emp['weekend_allowed'] = emp.get('weekend_work_allowed', True)
            emp['night_restriction'] = emp.get('night_shift_restriction', False)
            emp['seniority'] = emp.get('seniority_years', 0)
            emp['performance'] = emp.get('performance_rating', 3.0)
            emp['annual_vacation'] = emp.get('annual_vacation_days', 25)

        return employees

    except Error as e:
        st.error(f"Chyba pri načítavaní zamestnancov: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_safe_time_value(time_value, default_hour=9, default_minute=0):
    """Bezpečne konvertuje hodnotu na time objekt"""
    if isinstance(time_value, time):
        return time_value

    if time_value is None:
        return time(default_hour, default_minute)

    try:
        if isinstance(time_value, str):
            # Rôzne formáty stringov
            for fmt in ['%H:%M:%S', '%H:%M', '%H:%M:%S.%f']:
                try:
                    return datetime.strptime(time_value, fmt).time()
                except ValueError:
                    continue

        elif isinstance(time_value, timedelta):
            total_seconds = int(time_value.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return time(hours % 24, minutes)  # Zabezpečíme že hodiny sú 0-23

        elif hasattr(time_value, 'hour') and hasattr(time_value, 'minute'):
            # Datetime objekt
            return time_value.time()

    except Exception as e:
        print(f"⚠️ Chyba pri konverzii času {time_value}: {e}")

    # Fallback na default
    return time(default_hour, default_minute)


def load_shifts_from_db() -> List[Dict]:
    """Načíta smeny z databázy"""
    connection = get_db_connection()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM shift_types")
        shifts = cursor.fetchall()

        # Konvertujeme JSON stringy späť na objekty a upravíme názvy
        for shift in shifts:
            if shift.get('required_skills'):
                try:
                    shift['skills'] = json.loads(shift['required_skills'])
                except:
                    shift['skills'] = []
            else:
                shift['skills'] = []

            # Bezpečná konverzia časových údajov
            shift['start'] = get_safe_time_value(shift.get('start_time'), 9, 0)
            shift['end'] = get_safe_time_value(shift.get('end_time'), 17, 0)

            # Normalizácia skill level hodnôt
            min_skill_level = shift.get('min_skill_level', 'BEGINNER')
            level_mapping = {
                'BEGINNER': 'Začiatočník',
                'INTERMEDIATE': 'Pokročilý',
                'ADVANCED': 'Expert',
                'SUPERVISOR': 'Supervízor',
                'Za?iato?ník': 'Začiatočník',
                'Pokro?ilý': 'Pokročilý',
                'Zaciatok': 'Začiatočník',
                'Pokrocily': 'Pokročilý'
            }
            shift['min_level'] = level_mapping.get(min_skill_level, 'Začiatočník')

            # Ostatné mapovanie
            shift['rest_days'] = shift.get('rest_days_after', 0)
            shift['premium'] = shift.get('premium_pay', 0.0)
            shift['weekend_applicable'] = shift.get('is_weekend_applicable', True)
            shift['max_consecutive'] = shift.get('max_consecutive_days', 5)

        return shifts

    except Error as e:
        st.error(f"Chyba pri načítavaní smien: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def sync_to_database(st_instance) -> bool:
    """Synchronizuje všetky dáta do databázy"""
    try:
        success = True

        # Najprv sa uistíme, že databáza má správnu štruktúru
        print("🔄 Kontrolujem štruktúru databázy...")
        if not init_database():
            st_instance.error("Chyba pri inicializácii databázy")
            return False

        # Uloženie tímov
        if hasattr(st_instance, 'session_state') and 'teams' in st_instance.session_state:
            print(f"💾 Ukladám {len(st_instance.session_state.teams)} tímov...")
            if not save_teams_to_db(st_instance.session_state.teams):
                success = False

        # Uloženie zamestnancov
        if hasattr(st_instance, 'session_state') and 'employees' in st_instance.session_state:
            print(f"💾 Ukladám {len(st_instance.session_state.employees)} zamestnancov...")
            if not save_employees_to_db(st_instance.session_state.employees):
                success = False

        # Uloženie smien
        if hasattr(st_instance, 'session_state') and 'shifts' in st_instance.session_state:
            print(f"💾 Ukladám {len(st_instance.session_state.shifts)} smien...")
            if not save_shifts_to_db(st_instance.session_state.shifts):
                success = False

        if success:
            print("✅ Všetky dáta úspešne synchronizované!")
        else:
            print("⚠️ Niektoré dáta sa nepodarilo synchronizovať")

        return success

    except Exception as e:
        st_instance.error(f"Chyba pri synchronizácii: {e}")
        print(f"❌ Chyba pri synchronizácii: {e}")
        return False


def load_from_database(st_instance) -> bool:
    """Načíta všetky dáta z databázy"""
    try:
        print("🔄 Načítavam dáta z databázy...")

        # Načítanie tímov
        teams = load_teams_from_db()
        if teams:
            st_instance.session_state.teams = teams
            print(f"✅ Načítaných {len(teams)} tímov")
        else:
            print("⚠️ Žiadne tímy sa nenačítali")

        # Načítanie zamestnancov
        employees = load_employees_from_db()
        if employees:
            st_instance.session_state.employees = employees
            print(f"✅ Načítaných {len(employees)} zamestnancov")
        else:
            print("⚠️ Žiadni zamestnanci sa nenačítali")

        # Načítanie smien - s dodatočnou bezpečnostnou kontrolou
        shifts = load_shifts_from_db()
        if shifts:
            # Dvojitá kontrola časových údajov pre Streamlit
            for i, shift in enumerate(shifts):
                shifts[i]['start'] = get_safe_time_value(shift.get('start'), 9, 0)
                shifts[i]['end'] = get_safe_time_value(shift.get('end'), 17, 0)

            st_instance.session_state.shifts = shifts
            print(f"✅ Načítaných {len(shifts)} smien")
        else:
            print("⚠️ Žiadne smeny sa nenačítali")

        print("✅ Import z databázy dokončený")
        return True

    except Exception as e:
        st_instance.error(f"Chyba pri načítavaní z databázy: {e}")
        print(f"❌ Chyba pri načítavaní z databázy: {e}")
        return False


def add_database_controls(st_instance):
    """Pridá databázové kontroly do sidebar"""
    st_instance.subheader("🗄️ Databáza")

    if st_instance.button("🔄 Reinicializovať DB"):
        if init_database():
            st_instance.success("✅ Databáza reinicializovaná")
        else:
            st_instance.error("❌ Chyba pri reinicializácii")

    # Test spojenia
    connection = get_db_connection()
    if connection:
        st_instance.success("🔗 Databáza pripojená")
        connection.close()
    else:
        st_instance.error("❌ Problém s pripojením")


def save_generated_schedule(schedule_data: pd.DataFrame, period_start: date, period_end: date,
                            schedule_name: str = None) -> bool:
    """Uloží vygenerovaný plán do databázy"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Uloženie do schedule_history
        if schedule_name is None:
            schedule_name = f"Plán {period_start} - {period_end}"

        total_employees = schedule_data['Zamestnanec'].nunique() if not schedule_data.empty else 0
        total_hours = schedule_data['Hodiny'].sum() if not schedule_data.empty else 0
        total_cost = schedule_data['Náklady'].sum() if not schedule_data.empty else 0

        history_query = """
        INSERT INTO schedule_history (schedule_name, period_start, period_end, total_employees, total_hours, total_cost, schedule_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        schedule_json = schedule_data.to_json(orient='records', date_format='iso') if not schedule_data.empty else '{}'

        cursor.execute(history_query, (
            schedule_name, period_start, period_end, total_employees, total_hours, total_cost, schedule_json
        ))

        connection.commit()
        return True

    except Error as e:
        st.error(f"Chyba pri ukladaní plánu: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def load_existing_schedule(schedule_id: int) -> Optional[pd.DataFrame]:
    """Načíta existujúci plán z databázy"""
    connection = get_db_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM schedule_history WHERE id = %s", (schedule_id,))
        schedule = cursor.fetchone()

        if schedule and schedule.get('schedule_data'):
            import json
            schedule_data = json.loads(schedule['schedule_data'])
            return pd.DataFrame(schedule_data)

        return None

    except Error as e:
        st.error(f"Chyba pri načítavaní plánu: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


class DatabaseManager:
    """Hlavná trieda pre správu databázy"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Pripojí sa k databáze"""
        self.connection = get_db_connection()
        return self.connection is not None

    def disconnect(self):
        """Odpojí sa od databázy"""
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def test_connection(self) -> bool:
        """Otestuje spojenie s databázou"""
        test_conn = get_db_connection()
        if test_conn:
            test_conn.close()
            return True
        return False