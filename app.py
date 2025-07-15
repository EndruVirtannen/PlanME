# rn.py - Hlavná aplikácia s databázovou integráciou

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Sequence, Tuple, Optional, Set
from enum import Enum
import json
import calendar

import pandas as pd
from dateutil.rrule import rrule, DAILY
from ortools.sat.python import cp_model
import streamlit as st

try:
    from streamlit_calendar import date_picker  # type: ignore
except ImportError:
    date_picker = None

# Import databázového modulu
from database_manager import (
    DatabaseManager, init_database, sync_to_database,
    load_from_database, add_database_controls,
    save_generated_schedule, load_existing_schedule
)


class SkillLevel(Enum):
    BEGINNER = "Zaciatocník"
    INTERMEDIATE = "Pokrocilý"
    ADVANCED = "Expert"
    SUPERVISOR = "Supervízor"


class ContractType(Enum):
    FULL_TIME = "Plný úvazok"
    PART_TIME = "Ciastočný úvazok"
    TEMPORARY = "Docasný"
    CONTRACT = "Zmluvný"


class VacationType(Enum):
    ANNUAL = "Riadna dovolenka"
    SICK = "Nemocenská"
    PERSONAL = "Osobné volno"
    MATERNITY = "Materská/otcovská"
    UNPAID = "Neplatené voľno"
    TRAINING = "Skolenie"
    COMPENSATION = "Náhradné volno"


class Priority(Enum):
    LOW = "Nízka"
    MEDIUM = "Stredná"
    HIGH = "Vysoká"
    CRITICAL = "Kritická"


@dataclass
class Skill:
    name: str
    level: SkillLevel
    priority: int = 1  # 1 = najvyššia, 5 = najnižšia
    certification_expiry: Optional[date] = None


@dataclass
class VacationRequest:
    start_date: date
    end_date: date
    vacation_type: VacationType = VacationType.ANNUAL
    reason: str = ""
    approved: bool = False
    priority: Priority = Priority.MEDIUM
    created_date: date = field(default_factory=date.today)

    def get_duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def overlaps_with(self, other_start: date, other_end: date) -> bool:
        return not (self.end_date < other_start or self.start_date > other_end)


@dataclass
class ShiftType:
    name: str
    start: time
    end: time
    min_rest_hours_after: int = 11
    rest_days_after: int = 0
    required_skills: List[str] = field(default_factory=list)
    min_skill_level: SkillLevel = SkillLevel.BEGINNER
    difficulty_multiplier: float = 1.0
    premium_pay: float = 0.0
    is_weekend_applicable: bool = True
    max_consecutive_days: int = 7
    min_employees: int = 1
    max_employees: int = 10

    def duration_hours(self) -> float:
        start_dt = datetime.combine(date.today(), self.start)
        end_dt = datetime.combine(date.today(), self.end)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        return (end_dt - start_dt).total_seconds() / 3600


@dataclass
class WorkRequirement:
    monthly_hours_target: int = 160
    weekly_hours_min: int = 20
    weekly_hours_max: int = 48
    max_overtime_hours: int = 10
    min_days_off_per_week: int = 1
    max_consecutive_work_days: int = 6


@dataclass
class Employee:
    id: str
    name: str
    team_id: str
    max_consecutive_days: int = 5
    preferences: List[str] = field(default_factory=list)
    annual_vacation_days: int = 25
    max_night_shifts: int = 999
    contract_type: ContractType = ContractType.FULL_TIME
    skills: List[Skill] = field(default_factory=list)
    hourly_rate: float = 15.0
    seniority_years: int = 0
    can_work_alone: bool = True
    needs_supervision: bool = False
    email: str = ""
    phone: str = ""
    emergency_contact: str = ""
    notes: str = ""

    # Nové rozšírené atribúty
    vacation_requests: List[VacationRequest] = field(default_factory=list)
    work_requirements: WorkRequirement = field(default_factory=WorkRequirement)
    overtime_eligible: bool = True
    weekend_work_allowed: bool = True
    night_shift_restriction: bool = False
    start_date: Optional[date] = None
    probation_end_date: Optional[date] = None
    performance_rating: float = 3.0  # 1-5 škála
    languages: List[str] = field(default_factory=list)

    def is_available(self, d: date) -> bool:
        # Kontrola dovolenkových žiadostí
        for vacation in self.vacation_requests:
            if vacation.approved and vacation.start_date <= d <= vacation.end_date:
                return False
        return True

    def vacation_remaining(self) -> int:
        approved_annual = sum(
            vr.get_duration_days() for vr in self.vacation_requests
            if vr.approved and vr.vacation_type == VacationType.ANNUAL
        )
        return self.annual_vacation_days - approved_annual

    def has_skill(self, skill_name: str, min_level: SkillLevel = SkillLevel.BEGINNER) -> bool:
        for skill in self.skills:
            if skill.name == skill_name:
                skill_levels = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED,
                                SkillLevel.SUPERVISOR]
                return skill_levels.index(skill.level) >= skill_levels.index(min_level)
        return False

    def get_skill_level(self, skill_name: str) -> Optional[SkillLevel]:
        for skill in self.skills:
            if skill.name == skill_name:
                return skill.level
        return None

    def get_pending_vacation_days(self) -> int:
        return sum(
            vr.get_duration_days() for vr in self.vacation_requests
            if not vr.approved and vr.vacation_type == VacationType.ANNUAL
        )


@dataclass
class CompanyRequirements:
    min_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: min_people}
    max_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: max_people}
    target_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: target_people}
    weekend_multiplier: float = 1.0  # Násobiteľ pokrytia pre víkendy
    holiday_multiplier: float = 0.5  # Násobiteľ pokrytia pre sviatky
    emergency_contact_required: bool = True
    supervisor_always_present: bool = False


@dataclass
class Team:
    id: str
    name: str
    description: str = ""
    manager_id: Optional[str] = None
    priority: int = 1
    budget_limit: Optional[float] = None
    can_collaborate_with: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    color: str = "#4CAF50"

    # Nové pokročilé atribúty
    company_requirements: CompanyRequirements = field(default_factory=CompanyRequirements)
    department: str = ""
    cost_center: str = ""
    location: str = ""

    def get_employees(self, all_employees: List[Employee]) -> List[Employee]:
        return [emp for emp in all_employees if emp.team_id == self.id]


@dataclass
class Collaboration:
    team1_id: str
    team2_id: str
    shift_types: List[str] = field(default_factory=list)
    max_shared_employees: int = 2
    priority: int = 1


class AdvancedScheduler:
    def __init__(
            self,
            employees: Sequence[Employee],
            teams: Sequence[Team],
            shift_types: Sequence[ShiftType],
            period_start: date,
            period_end: date,
            coverage: Dict[str, Dict[str, int]] | None = None,
            collaborations: List[Collaboration] = None,
            max_total_hours_per_employee: int | None = None,
            consider_skills: bool = True,
            balance_workload: bool = True,
            minimize_cost: bool = False,
            company_requirements: CompanyRequirements = None
    ) -> None:
        self.employees = list(employees)
        self.teams = list(teams)
        self.shift_types = list(shift_types)
        self.period_start = period_start
        self.period_end = period_end
        self.coverage = coverage or {}
        self.collaborations = collaborations or []
        self.max_total_hours_per_employee = max_total_hours_per_employee
        self.consider_skills = consider_skills
        self.balance_workload = balance_workload
        self.minimize_cost = minimize_cost
        self.company_requirements = company_requirements or CompanyRequirements()

        self._dates = list(rrule(DAILY, dtstart=period_start, until=period_end))
        self.model = cp_model.CpModel()
        self._build_vars()
        self._add_constraints()
        self._set_objective()

    def _build_vars(self) -> None:
        self.x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
        for e in range(len(self.employees)):
            for d in range(len(self._dates)):
                for s in range(len(self.shift_types)):
                    self.x[(e, d, s)] = self.model.NewBoolVar(f"x_{e}_{d}_{s}")

    def _is_weekend(self, date_obj: date) -> bool:
        return date_obj.weekday() >= 5  # Sobota = 5, Nedeľa = 6

    def _add_constraints(self) -> None:
        ne, nd, ns = len(self.employees), len(self._dates), len(self.shift_types)

        # ZÁKLADNÉ POKRYTIE - ZJEDNODUŠENÉ
        for d in range(nd):
            for s_idx, shift in enumerate(self.shift_types):
                # Jednoduché pokrytie - aspoň minimálne požiadavky
                total_required = 0
                for team in self.teams:
                    base_required = self.coverage.get(team.id, {}).get(shift.name, 0)
                    total_required += base_required

                if total_required > 0:
                    assigned_to_shift = sum(self.x[(e, d, s_idx)] for e in range(ne))
                    # Iba minimálne pokrytie, bez maxím
                    self.model.Add(assigned_to_shift >= max(1, total_required))

        # Každý zamestnanec max. jedna smena za deň
        for e in range(ne):
            for d in range(nd):
                self.model.Add(sum(self.x[(e, d, s)] for s in range(ns)) <= 1)

        # ZJEDNODUŠENÉ PRACOVNÉ POŽIADAVKY
        for e_idx, emp in enumerate(self.employees):
            # Dostupnosť (dovolenky) - JEDINÉ TVRDÉ OBMEDZENIE
            for d_idx, dt in enumerate(self._dates):
                if not emp.is_available(dt.date()):
                    for s in range(ns):
                        self.model.Add(self.x[(e_idx, d_idx, s)] == 0)

            # Preferencie - IBA AK SÚ DEFINOVANÉ
            if emp.preferences:
                allowed_shifts = {
                    s for s in range(ns) if self.shift_types[s].name in emp.preferences
                }
                if allowed_shifts and len(allowed_shifts) < ns:  # Iba ak nie sú všetky smeny povolené
                    for d_idx, dt in enumerate(self._dates):
                        if emp.is_available(dt.date()):
                            for s in range(ns):
                                if s not in allowed_shifts:
                                    self.model.Add(self.x[(e_idx, d_idx, s)] == 0)

            # MÄKKÉ LIMITY - iba ak je balance_workload True
            if self.balance_workload:
                # Maximálne po sebe idúce dni - VEĽMI JEDNODUCHÉ
                max_consecutive = min(emp.work_requirements.max_consecutive_work_days, 7)
                if max_consecutive < 7:  # Iba ak je nastavené
                    for start in range(nd - max_consecutive):
                        if start + max_consecutive < nd:
                            consecutive_work = sum(
                                sum(self.x[(e_idx, start + i, s)] for s in range(ns))
                                for i in range(max_consecutive + 1)
                            )
                            self.model.Add(consecutive_work <= max_consecutive)

                # Týždenné limity - VEĽMI VOĽNÉ
                weeks = math.ceil(nd / 7)
                for week in range(weeks):
                    week_start = week * 7
                    week_end = min(week_start + 7, nd)

                    work_days = sum(
                        sum(self.x[(e_idx, d, s)] for s in range(ns))
                        for d in range(week_start, week_end)
                    )

                    # Maximálne 6 dní v týždni
                    max_work_days = min(week_end - week_start, 6)
                    self.model.Add(work_days <= max_work_days)

        # Skillové požiadavky - IBA AK SÚ KRITICKÉ
        if self.consider_skills:
            for s_idx, shift in enumerate(self.shift_types):
                if shift.required_skills:
                    for d in range(nd):
                        for e_idx, emp in enumerate(self.employees):
                            # Iba ak zamestnancovi úplne chýbajú potrebné skills
                            missing_critical_skills = [
                                skill for skill in shift.required_skills
                                if not emp.has_skill(skill, SkillLevel.BEGINNER)
                            ]
                            if missing_critical_skills:
                                self.model.Add(self.x[(e_idx, d, s_idx)] == 0)

        # Odpočinok po nočných smenách - IBA PRE NOČNÉ
        for s_idx, shift in enumerate(self.shift_types):
            if shift.rest_days_after > 0 and "nočná" in shift.name.lower():
                for e_idx in range(ne):
                    for d_idx in range(nd - 1):  # Iba jeden deň odpočinku
                        if d_idx + 1 < nd:
                            for s2 in range(ns):
                                self.model.Add(
                                    self.x[(e_idx, d_idx, s_idx)] + self.x[(e_idx, d_idx + 1, s2)] <= 1
                                )

        # Nočné smeny - IBA ÚPLNÝ ZÁKAZ
        for s_idx, shift in enumerate(self.shift_types):
            if "nočná" in shift.name.lower():
                for e_idx, emp in enumerate(self.employees):
                    if emp.night_shift_restriction:
                        for d in range(nd):
                            self.model.Add(self.x[(e_idx, d, s_idx)] == 0)

    def _set_objective(self) -> None:
        ne, nd, ns = len(self.employees), len(self._dates), len(self.shift_types)

        if self.minimize_cost:
            # Jednoduchá minimalizácia nákladov
            total_cost = []
            for e_idx, emp in enumerate(self.employees):
                for d in range(nd):
                    for s_idx, shift in enumerate(self.shift_types):
                        base_cost = int(emp.hourly_rate * shift.duration_hours() * 100)
                        total_cost.append(self.x[(e_idx, d, s_idx)] * base_cost)
            self.model.Minimize(sum(total_cost))

        elif self.balance_workload:
            # Jednoduché vyváženie záťaže
            max_shifts = self.model.NewIntVar(0, nd, "max_shifts")
            min_shifts = self.model.NewIntVar(0, nd, "min_shifts")

            for e in range(ne):
                total_shifts = sum(self.x[(e, d, s)] for d in range(nd) for s in range(ns))
                self.model.Add(total_shifts <= max_shifts)
                self.model.Add(total_shifts >= min_shifts)

            self.model.Minimize(max_shifts - min_shifts)

        else:
            # Maximalizovať celkové priradenie smien (jednoduchý fallback)
            total_assignments = sum(
                self.x[(e, d, s)] for e in range(ne) for d in range(nd) for s in range(ns)
            )
            self.model.Maximize(total_assignments)

    def solve(self, limit: int = 180) -> pd.DataFrame:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        status = solver.Solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                "Riešenie sa nenašlo v čase limitu. Skúste znížiť požiadavky alebo pridať viac zamestnancov.")

        rows: List[Dict[str, object]] = []
        for d_idx, dt_rule in enumerate(self._dates):
            dt = dt_rule.date()
            for s_idx, stype in enumerate(self.shift_types):
                for e_idx, emp in enumerate(self.employees):
                    if solver.BooleanValue(self.x[(e_idx, d_idx, s_idx)]):
                        team = next((t for t in self.teams if t.id == emp.team_id), None)

                        # Výpočet nákladov s bonusmi
                        is_weekend = self._is_weekend(dt)
                        base_cost = emp.hourly_rate * stype.duration_hours()
                        weekend_bonus = 1.2 if is_weekend else 1.0
                        shift_premium = 1 + stype.premium_pay
                        final_cost = base_cost * weekend_bonus * shift_premium

                        rows.append({
                            "Dátum": dt,
                            "Zmena": stype.name,
                            "Zamestnanec": emp.name,
                            "ID": emp.id,
                            "Tím": team.name if team else "Neznámy",
                            "Hodiny": stype.duration_hours(),
                            "Náklady": final_cost,
                            "Je_víkend": is_weekend,
                            "Víkendový_bonus": weekend_bonus,
                            "Prémia_smeny": shift_premium
                        })
        return pd.DataFrame(rows)

    def get_summary(self, schedule: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if not schedule.empty:
            print(f"Dostupné stĺpce v schedule_df: {list(schedule.columns)}")
            print(f"Počet riadkov: {len(schedule)}")

        # Rozšírený súhrn pre zamestnancov
        employee_summary = []
        for emp in self.employees:
            if "Zamestnanec" in schedule.columns:
                emp_data = schedule[schedule["Zamestnanec"] == emp.name]
            else:
                emp_data = schedule.iloc[0:0]

            total_hours = emp_data["Hodiny"].sum() if "Hodiny" in emp_data.columns else 0
            total_cost = emp_data["Náklady"].sum() if "Náklady" in emp_data.columns else 0
            weekend_hours = emp_data[emp_data["Je_víkend"] == True][
                "Hodiny"].sum() if "Je_víkend" in emp_data.columns else 0
            shift_counts = emp_data[
                "Zmena"].value_counts().to_dict() if "Zmena" in emp_data.columns and not emp_data.empty else {}

            # Cieľové vs skutočné hodiny
            target_hours = emp.work_requirements.monthly_hours_target
            hours_diff = total_hours - target_hours
            hours_status = "✅ V cieli" if abs(hours_diff) <= 20 else "⚠️ Mimo cieľa"

            employee_summary.append({
                "ID": emp.id,
                "Zamestnanec": emp.name,
                "Tím": next((t.name for t in self.teams if t.id == emp.team_id), "Neznámy"),
                "Odpracované hodiny": total_hours,
                "Cieľové hodiny": target_hours,
                "Rozdiel": hours_diff,
                "Status": hours_status,
                "Víkendové hodiny": weekend_hours,
                "Celkom nákladov": total_cost,
                "Zostatok dovolenky": emp.vacation_remaining(),
                "Čakajúce žiadosti": emp.get_pending_vacation_days(),
                **{f"Smeny {k}": v for k, v in shift_counts.items()}
            })

        # Súhrn pre tímy
        team_summary = []
        for team in self.teams:
            if "Tím" in schedule.columns:
                team_data = schedule[schedule["Tím"] == team.name]
            else:
                team_data = schedule.iloc[0:0]

            total_hours = team_data["Hodiny"].sum() if "Hodiny" in team_data.columns else 0
            total_cost = team_data["Náklady"].sum() if "Náklady" in team_data.columns else 0
            employee_count = len(
                team_data["Zamestnanec"].unique()) if "Zamestnanec" in team_data.columns and not team_data.empty else 0
            weekend_cost = team_data[team_data["Je_víkend"] == True][
                "Náklady"].sum() if "Je_víkend" in team_data.columns else 0

            team_summary.append({
                "Tím": team.name,
                "Zamestnanci": employee_count,
                "Celkom hodín": total_hours,
                "Priemerné hodiny/zamestnanec": total_hours / employee_count if employee_count > 0 else 0,
                "Celkom nákladov": total_cost,
                "Víkendové náklady": weekend_cost,
                "Priemerné náklady/zamestnanec": total_cost / employee_count if employee_count > 0 else 0
            })

        # Analýza dovoleniek
        vacation_summary = []
        for emp in self.employees:
            pending_requests = [vr for vr in emp.vacation_requests if not vr.approved]
            approved_requests = [vr for vr in emp.vacation_requests if vr.approved]

            vacation_summary.append({
                "Zamestnanec": emp.name,
                "Ročný nárok": emp.annual_vacation_days,
                "Využité dni": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.ANNUAL),
                "Zostatok": emp.vacation_remaining(),
                "Čakajúce žiadosti": len(pending_requests),
                "Čakajúce dni": sum(
                    vr.get_duration_days() for vr in pending_requests if vr.vacation_type == VacationType.ANNUAL),
                "Nemocenské dni": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.SICK),
                "Osobné voľno": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.PERSONAL)
            })

        return {
            "employees": pd.DataFrame(employee_summary),
            "teams": pd.DataFrame(team_summary),
            "vacations": pd.DataFrame(vacation_summary)
        }


# Inicializácia databázy pri prvom spustení
if 'db_initialized' not in st.session_state:
    if init_database():
        st.session_state.db_initialized = True
    else:
        st.error("❌ Chyba pri inicializácii databázy")

# Streamlit UI - Pokročilá verzia s databázou
st.set_page_config(page_title="PlanME Pro – Enterprise Scheduler", page_icon="🏢", layout="wide")
st.title("🏢 PlanME Pro – Enterprise Team Scheduler")

# Inicializácia premenných pre neskoršie použitie
start_date = date.today()
end_date = date.today() + timedelta(days=30)
employees = []
teams = []
shift_types = []
collaborations = []
schedule_df = pd.DataFrame()

# Sidebar pre globálne nastavenia
with st.sidebar:
    st.header("⚙️ Globálne nastavenia")
    consider_skills = st.checkbox("Zohľadniť zručnosti", value=True)
    balance_workload = st.checkbox("Vyvážiť pracovnú záťaž", value=True)
    minimize_cost = st.checkbox("Minimalizovať náklady", value=False)

    st.header("🎯 Optimalizačné ciele")
    optimization_goal = st.selectbox(
        "Hlavný cieľ optimalizácie",
        ["Vyváženie záťaže", "Minimalizácia nákladov", "Maximalizácia spokojnosti", "Splnenie cieľových hodín"]
    )

    # PRIDANÉ: Databázové kontroly
    add_database_controls(st)

    st.header("📊 Export/Import")

    # Modifikované tlačidlá pre import/export s databázou
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Import z DB"):
            with st.spinner("Načítavam z databázy..."):
                if load_from_database(st):
                    st.success("✅ Dáta načítané")
                    st.rerun()

    with col2:
        if st.button("📤 Export do DB"):
            with st.spinner("Ukladám do databázy..."):
                if sync_to_database(st):
                    st.success("✅ Dáta uložené")

# Hlavné tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "⏰ Obdobie & Smeny",
    "🏢 Tímy & Požiadavky",
    "👥 Zamestnanci",
    "🏖️ Dovolenky",
    "🤝 Spolupráca",
    "📊 Generovanie",
    "📈 Analýzy",
    "💾 Databáza"
])
with tab1:
    st.subheader("📅 Plánovacie obdobie")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Začiatok", date.today())
    with col2:
        end_date = st.date_input("Koniec", date.today() + timedelta(days=30))

    if end_date < start_date:
        st.error("Koniec nesmie byť pred začiatkom!")
        st.stop()

    # Počet dní a základné info
    total_days = (end_date - start_date).days + 1
    weekdays = sum(1 for d in range(total_days) if (start_date + timedelta(d)).weekday() < 5)
    weekends = total_days - weekdays

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Celkom dní", total_days)
    with col2:
        st.metric("Pracovné dni", weekdays)
    with col3:
        st.metric("Víkendové dni", weekends)

    st.subheader("⏰ Definícia smien")

    # Inicializácia session state pre smeny
    if 'shifts' not in st.session_state:
        st.session_state.shifts = [
            {
                "name": "Denná", "start": time(6), "end": time(14), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.0,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            },
            {
                "name": "Poobedná", "start": time(14), "end": time(22), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.1,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            },
            {
                "name": "Nočná", "start": time(22), "end": time(6), "rest_days": 1,
                "skills": ["Bezpečnosť"], "min_level": "Pokročilý", "premium": 0.25,
                "weekend_applicable": True, "max_consecutive": 3, "min_employees": 1, "max_employees": 2
            }
        ]

    # Náhrada pre riadky 660-690 v rn.py (tab1 - smeny sekcia)

    # Zabezpečenie spätnej kompatibility - pridanie chýbajúcich kľúčov
    for i, shift_data in enumerate(st.session_state.shifts):
        # Pridanie chýbajúcich kľúčov s defaultnými hodnotami
        default_values = {
            "weekend_applicable": True,
            "max_consecutive": 5,
            "min_employees": 1,
            "max_employees": 3,
            "premium": 0.0,
            "rest_days": 0,
            "skills": [],
            "min_level": "Zaciatocník"
        }

        for key, default_value in default_values.items():
            if key not in shift_data:
                st.session_state.shifts[i][key] = default_value

    # Správa smien
    shift_types = []
    for i, shift_data in enumerate(st.session_state.shifts):
        with st.expander(f"Smena: {shift_data['name']}", expanded=i == 0):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                name = st.text_input("Názov", shift_data['name'], key=f"shift_name_{i}")

                # Bezpečná konverzia time hodnôt
                start_val = shift_data.get('start')
                if not isinstance(start_val, time):
                    if start_val is None:
                        start_val = time(6, 0)
                    elif isinstance(start_val, str):
                        try:
                            start_val = datetime.strptime(start_val, '%H:%M:%S').time()
                        except:
                            start_val = time(6, 0)
                    elif isinstance(start_val, timedelta):
                        total_seconds = int(start_val.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        start_val = time(hours % 24, minutes)
                    else:
                        start_val = time(6, 0)

                end_val = shift_data.get('end')
                if not isinstance(end_val, time):
                    if end_val is None:
                        end_val = time(14, 0)
                    elif isinstance(end_val, str):
                        try:
                            end_val = datetime.strptime(end_val, '%H:%M:%S').time()
                        except:
                            end_val = time(14, 0)
                    elif isinstance(end_val, timedelta):
                        total_seconds = int(end_val.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        end_val = time(hours % 24, minutes)
                    else:
                        end_val = time(14, 0)

                start_time = st.time_input("Začiatok", start_val, key=f"shift_start_{i}")
                end_time = st.time_input("Koniec", end_val, key=f"shift_end_{i}")

            with col2:
                rest_days = st.number_input("Dni voľna po smene", 0, 7, int(shift_data['rest_days']),
                                            key=f"shift_rest_{i}")

                # Oprava pre mixed numeric types
                premium_val = float(shift_data.get('premium', 0.0))
                premium = st.number_input("Prémia (%)", 0.0, 1.0, premium_val, step=0.05,
                                          key=f"shift_premium_{i}")
                weekend_applicable = st.checkbox("Platí aj na víkendy", shift_data['weekend_applicable'],
                                                 key=f"shift_weekend_{i}")

            with col3:
                min_employees = st.number_input("Min. zamestnancov", 1, 10, int(shift_data['min_employees']),
                                                key=f"shift_min_{i}")
                max_employees = st.number_input("Max. zamestnancov", 1, 20, int(shift_data['max_employees']),
                                                key=f"shift_max_{i}")
                max_consecutive = st.number_input("Max. po sebe idúcich dní", 1, 14, int(shift_data['max_consecutive']),
                                                  key=f"shift_consec_{i}")

            with col4:
                skills = st.multiselect("Požadované zručnosti",
                                        ["Prvá pomoc", "Vedenie tímu", "Technické zručnosti", "Komunikácia",
                                         "Bezpečnosť"],
                                        shift_data['skills'], key=f"shift_skills_{i}")
                min_level = st.selectbox("Min. úroveň",
                                         ["Zaciatocník", "Pokročilý", "Expert", "Supervízor"],
                                         index=["Zaciatocník", "Pokročilý", "Expert", "Supervízor"].index(
                                             shift_data['min_level']),
                                         key=f"shift_level_{i}")

            # Aktualizácia session state
            st.session_state.shifts[i] = {
                "name": name, "start": start_time, "end": end_time, "rest_days": rest_days,
                "skills": skills, "min_level": min_level, "premium": premium,
                "weekend_applicable": weekend_applicable, "max_consecutive": max_consecutive,
                "min_employees": min_employees, "max_employees": max_employees
            }

            shift_types.append(ShiftType(
                name=name, start=start_time, end=end_time, rest_days_after=rest_days,
                required_skills=skills, min_skill_level=SkillLevel(min_level), premium_pay=premium,
                is_weekend_applicable=weekend_applicable, max_consecutive_days=max_consecutive,
                min_employees=min_employees, max_employees=max_employees
            ))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať smenu"):
            st.session_state.shifts.append({
                "name": "Nová smena", "start": time(9), "end": time(17), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.0,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť poslednú smenu") and len(st.session_state.shifts) > 1:
            st.session_state.shifts.pop()
            st.rerun()

with tab2:
    st.subheader("🏢 Správa tímov a firemných požiadaviek")

    # Inicializácia session state pre tímy
    if 'teams' not in st.session_state:
        st.session_state.teams = [
            {
                "id": "TEAM001", "name": "Prevádzkový tím", "description": "Hlavný prevádzkový tím",
                "priority": 1, "color": "#4CAF50", "department": "Výroba", "location": "Bratislava",
                "min_coverage": {"Denná": 2, "Poobedná": 2, "Nočná": 1},
                "max_coverage": {"Denná": 4, "Poobedná": 4, "Nočná": 2},
                "target_coverage": {"Denná": 3, "Poobedná": 3, "Nočná": 1},
                "weekend_multiplier": 1.0, "holiday_multiplier": 0.5,
                "supervisor_required": False, "emergency_contact": True
            }
        ]

    # Zabezpečenie spätnej kompatibility pre tímy
    for i, team_data in enumerate(st.session_state.teams):
        default_team_values = {
            "department": "",
            "location": "",
            "min_coverage": {},
            "max_coverage": {},
            "target_coverage": {},
            "weekend_multiplier": 1.0,
            "holiday_multiplier": 0.5,
            "supervisor_required": False,
            "emergency_contact": True
        }

        for key, default_value in default_team_values.items():
            if key not in team_data:
                st.session_state.teams[i][key] = default_value

    teams = []
    for i, team_data in enumerate(st.session_state.teams):
        with st.expander(f"Tím: {team_data['name']}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                team_id = st.text_input("ID tímu", team_data['id'], key=f"team_id_{i}")
                name = st.text_input("Názov tímu", team_data['name'], key=f"team_name_{i}")
                description = st.text_area("Popis", team_data['description'], key=f"team_desc_{i}")
                department = st.text_input("Oddelenie", team_data.get('department', ''), key=f"team_dept_{i}")
                location = st.text_input("Lokalita", team_data.get('location', ''), key=f"team_loc_{i}")

            with col2:
                priority = st.number_input("Priorita", 1, 10, team_data['priority'], key=f"team_priority_{i}")
                color = st.color_picker("Farba", team_data['color'], key=f"team_color_{i}")

            # Firemné požiadavky na pokrytie
            st.write("**Požiadavky na pokrytie smien:**")
            col1, col2, col3 = st.columns(3)

            min_coverage = {}
            max_coverage = {}
            target_coverage = {}

            with col1:
                st.write("**Minimum:**")
                for shift in shift_types:
                    min_val = st.number_input(
                        f"Min {shift.name}", 0, 10,
                        team_data.get('min_coverage', {}).get(shift.name, 1),
                        key=f"team_min_{i}_{shift.name}"
                    )
                    min_coverage[shift.name] = min_val

            with col2:
                st.write("**Cieľ:**")
                for shift in shift_types:
                    target_val = st.number_input(
                        f"Cieľ {shift.name}", 0, 15,
                        team_data.get('target_coverage', {}).get(shift.name, 1),
                        key=f"team_target_{i}_{shift.name}"
                    )
                    target_coverage[shift.name] = target_val

            with col3:
                st.write("**Maximum:**")
                for shift in shift_types:
                    max_val = st.number_input(
                        f"Max {shift.name}", 0, 20,
                        team_data.get('max_coverage', {}).get(shift.name, 2),
                        key=f"team_max_{i}_{shift.name}"
                    )
                    max_coverage[shift.name] = max_val

            # Pokročilé nastavenia
            with st.expander("Pokročilé nastavenia tímu"):
                weekend_multiplier = st.number_input("Víkendový násobiteľ pokrytia", 0.1, 2.0, 1.0, step=0.1,
                                                     key=f"team_weekend_{i}")
                holiday_multiplier = st.number_input("Sviatkový násobiteľ pokrytia", 0.1, 2.0, 0.5, step=0.1,
                                                     key=f"team_holiday_{i}")
                supervisor_required = st.checkbox("Vždy vyžadovať supervízora", key=f"team_supervisor_{i}")
                emergency_contact = st.checkbox("Vyžadovať pohotovostný kontakt", True, key=f"team_emergency_{i}")

            # Aktualizácia session state
            st.session_state.teams[i] = {
                "id": team_id, "name": name, "description": description,
                "priority": priority, "color": color, "department": department, "location": location,
                "min_coverage": min_coverage, "max_coverage": max_coverage, "target_coverage": target_coverage,
                "weekend_multiplier": weekend_multiplier, "holiday_multiplier": holiday_multiplier,
                "supervisor_required": supervisor_required, "emergency_contact": emergency_contact
            }

            # Vytvorenie Company Requirements
            company_req = CompanyRequirements(
                min_coverage_per_shift=min_coverage,
                max_coverage_per_shift=max_coverage,
                target_coverage_per_shift=target_coverage,
                weekend_multiplier=weekend_multiplier,
                holiday_multiplier=holiday_multiplier,
                supervisor_always_present=supervisor_required,
                emergency_contact_required=emergency_contact
            )

            teams.append(Team(
                id=team_id, name=name, description=description, priority=priority,
                color=color, department=department, location=location,
                company_requirements=company_req
            ))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať tím"):
            new_id = f"TEAM{len(st.session_state.teams) + 1:03d}"
            st.session_state.teams.append({
                "id": new_id, "name": "Nový tím", "description": "", "priority": 1, "color": "#FF9800",
                "department": "", "location": "", "min_coverage": {}, "max_coverage": {}, "target_coverage": {}
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť posledný tím") and len(st.session_state.teams) > 1:
            st.session_state.teams.pop()
            st.rerun()

with tab3:
    st.subheader("👥 Správa zamestnancov")

    # Inicializácia session state pre zamestnancov
    if 'employees' not in st.session_state:
        st.session_state.employees = [
            {
                "id": "EMP001", "name": "Ján Novák", "team_id": "TEAM001", "max_cons": 5, "max_night": 8,
                "hourly_rate": 15.0, "skills": [], "monthly_target": 160, "weekly_min": 20, "weekly_max": 48,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 4.0, "seniority": 2, "email": "", "phone": "", "preferences": []
            },
            {
                "id": "EMP002", "name": "Mária Svobodová", "team_id": "TEAM001", "max_cons": 4, "max_night": 6,
                "hourly_rate": 18.0, "skills": [], "monthly_target": 160, "weekly_min": 30, "weekly_max": 45,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 4.5, "seniority": 5, "email": "", "phone": "", "preferences": []
            },
            {
                "id": "EMP003", "name": "Peter Kováč", "team_id": "TEAM001", "max_cons": 6, "max_night": 10,
                "hourly_rate": 20.0, "skills": [], "monthly_target": 170, "weekly_min": 25, "weekly_max": 50,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 3.5, "seniority": 1, "email": "", "phone": "", "preferences": []
            }
        ]

    # Náhrada pre tab3 (zamestnanci) v rn.py - oprava numeric types

    # Zabezpečenie spätnej kompatibility pre zamestnancov
    for i, emp_data in enumerate(st.session_state.employees):
        default_emp_values = {
            "monthly_target": 160,
            "weekly_min": 20,
            "weekly_max": 48,
            "overtime_eligible": True,
            "weekend_allowed": True,
            "night_restriction": False,
            "performance": 3.0,
            "seniority": 0,
            "email": "",
            "phone": "",
            "preferences": [],
            "skills": []
        }

        for key, default_value in default_emp_values.items():
            if key not in emp_data:
                st.session_state.employees[i][key] = default_value

    employees = []
    team_options = {team["id"]: team["name"] for team in st.session_state.teams}

    for i, emp_data in enumerate(st.session_state.employees):
        with st.expander(f"Zamestnanec: {emp_data['name']}", expanded=False):

            # Základné informácie
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Základné údaje:**")
                emp_id = st.text_input("ID", emp_data['id'], key=f"emp_id_{i}")
                name = st.text_input("Meno a priezvisko", emp_data['name'], key=f"emp_name_{i}")
                team_id = st.selectbox("Tím", list(team_options.keys()),
                                       index=list(team_options.keys()).index(emp_data['team_id']) if emp_data[
                                                                                                         'team_id'] in team_options else 0,
                                       format_func=lambda x: team_options[x], key=f"emp_team_{i}")
                email = st.text_input("Email", emp_data.get('email', ''), key=f"emp_email_{i}")
                phone = st.text_input("Telefón", emp_data.get('phone', ''), key=f"emp_phone_{i}")

            with col2:
                st.write("**Pracovné podmienky:**")
                monthly_target = st.number_input("Mesačný cieľ hodín", 80, 200,
                                                 int(emp_data.get('monthly_target', 160)),
                                                 key=f"emp_monthly_{i}")
                weekly_min = st.number_input("Min. týždenných hodín", 10, 40,
                                             int(emp_data.get('weekly_min', 20)),
                                             key=f"emp_weekly_min_{i}")
                weekly_max = st.number_input("Max. týždenných hodín", 30, 60,
                                             int(emp_data.get('weekly_max', 48)),
                                             key=f"emp_weekly_max_{i}")
                max_cons = st.number_input("Max. po sebe idúcich dní", 1, 14,
                                           int(emp_data.get('max_cons', 5)),
                                           key=f"emp_cons_{i}")
                max_night = st.number_input("Max. nočných smien", 0, 20,
                                            int(emp_data.get('max_night', 8)),
                                            key=f"emp_night_{i}")

            with col3:
                st.write("**Finančné a osobné:**")
                hourly_rate = st.number_input("Hodinová sadzba (€)", 10.0, 100.0,
                                              float(emp_data.get('hourly_rate', 15.0)),
                                              step=0.5, key=f"emp_rate_{i}")
                performance = st.number_input("Hodnotenie výkonu (1-5)", 1.0, 5.0,
                                              float(emp_data.get('performance', 3.0)),
                                              step=0.5, key=f"emp_perf_{i}")
                seniority = st.number_input("Roky stáže", 0, 40,
                                            int(emp_data.get('seniority', 0)),
                                            key=f"emp_senior_{i}")
                annual_vacation = st.number_input("Ročný nárok dovolenky", 20, 35, 25,
                                                  key=f"emp_vacation_{i}")

            # Obmedzenia a možnosti
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pracovné možnosti:**")
                contract_type = st.selectbox("Typ zmluvy",
                                             ["Plný úväzok", "Čiastočný úväzok", "Dočasný", "Zmluvný"],
                                             key=f"emp_contract_{i}")
                overtime_eligible = st.checkbox("Môže robiť nadčasy",
                                                bool(emp_data.get('overtime_eligible', True)),
                                                key=f"emp_overtime_{i}")
                weekend_allowed = st.checkbox("Môže pracovať cez víkend",
                                              bool(emp_data.get('weekend_allowed', True)),
                                              key=f"emp_weekend_{i}")
                night_restriction = st.checkbox("Zákaz nočných smien",
                                                bool(emp_data.get('night_restriction', False)),
                                                key=f"emp_night_restrict_{i}")

            with col2:
                st.write("**Zručnosti:**")
                available_skills = ["Prvá pomoc", "Vedenie tímu", "Technické zručnosti", "Komunikácia", "Bezpečnosť",
                                    "Jazykové", "IT"]
                employee_skills = []
                for skill_name in available_skills:
                    if st.checkbox(f"{skill_name}", key=f"emp_skill_{i}_{skill_name}"):
                        level = st.selectbox(f"Úroveň {skill_name}",
                                             ["Zaciatocník", "Pokročilý", "Expert", "Supervízor"],
                                             key=f"emp_skill_level_{i}_{skill_name}")
                        employee_skills.append(Skill(name=skill_name, level=SkillLevel(level)))

            # Preferencie smien
            st.write("**Preferencie smien:**")
            shift_names = [s["name"] for s in st.session_state.shifts]
            preferences = st.multiselect("Preferované smeny (prázdne = všetky)",
                                         shift_names,
                                         emp_data.get('preferences', []),
                                         key=f"emp_prefs_{i}")

            # Aktualizácia session state
            st.session_state.employees[i] = {
                "id": emp_id, "name": name, "team_id": team_id, "max_cons": max_cons,
                "max_night": max_night, "hourly_rate": hourly_rate, "skills": employee_skills,
                "monthly_target": monthly_target, "weekly_min": weekly_min, "weekly_max": weekly_max,
                "overtime_eligible": overtime_eligible, "weekend_allowed": weekend_allowed,
                "night_restriction": night_restriction, "performance": performance, "seniority": seniority,
                "email": email, "phone": phone, "preferences": preferences
            }

            # Vytvorenie Work Requirements
            work_req = WorkRequirement(
                monthly_hours_target=monthly_target,
                weekly_hours_min=weekly_min,
                weekly_hours_max=weekly_max,
                max_consecutive_work_days=max_cons
            )

            employees.append(Employee(
                id=emp_id, name=name, team_id=team_id, max_consecutive_days=max_cons,
                max_night_shifts=max_night, hourly_rate=hourly_rate, skills=employee_skills,
                contract_type=ContractType(contract_type), work_requirements=work_req,
                overtime_eligible=overtime_eligible, weekend_work_allowed=weekend_allowed,
                night_shift_restriction=night_restriction, seniority_years=seniority,
                performance_rating=performance, email=email, phone=phone,
                preferences=preferences, annual_vacation_days=annual_vacation
            ))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať zamestnanca"):
            new_id = f"EMP{len(st.session_state.employees) + 1:03d}"
            st.session_state.employees.append({
                "id": new_id, "name": "Nový zamestnanec",
                "team_id": list(team_options.keys())[0] if team_options else "TEAM001",
                "max_cons": 5, "max_night": 8, "hourly_rate": 15.0, "skills": [],
                "monthly_target": 160, "weekly_min": 20, "weekly_max": 48,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 3.0, "seniority": 0, "email": "", "phone": "", "preferences": []
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť posledného"):
            if len(st.session_state.employees) > 1:
                st.session_state.employees.pop()
                st.rerun()

with tab4:
    st.subheader("🏖️ Správa dovoleniek a neprítomností")

    # Inicializácia session state pre dovolenky
    if 'vacation_requests' not in st.session_state:
        st.session_state.vacation_requests = {}

    # Výber zamestnanca pre správu dovolenky
    employee_names = {emp["id"]: emp["name"] for emp in st.session_state.employees}
    selected_emp_id = st.selectbox("Vyberte zamestnanca:", list(employee_names.keys()),
                                   format_func=lambda x: employee_names[x])

    if selected_emp_id:
        selected_emp = next(emp for emp in st.session_state.employees if emp["id"] == selected_emp_id)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Dovolenka pre: {selected_emp['name']}**")

            # Inicializácia dovoleniek pre zamestnanca
            if selected_emp_id not in st.session_state.vacation_requests:
                st.session_state.vacation_requests[selected_emp_id] = []

            # Nová žiadosť o dovolenku
            with st.expander("➕ Nová žiadosť o dovolenku", expanded=True):
                vacation_start = st.date_input("Začiatok", key=f"vac_start_{selected_emp_id}")
                vacation_end = st.date_input("Koniec", key=f"vac_end_{selected_emp_id}")
                vacation_type = st.selectbox("Typ neprítomnosti",
                                             ["Riadna dovolenka", "Nemocenská", "Osobné voľno", "Materská/otcovská",
                                              "Neplatené voľno", "Školenie", "Náhradné voľno"],
                                             key=f"vac_type_{selected_emp_id}")
                vacation_reason = st.text_area("Dôvod/Poznámka", key=f"vac_reason_{selected_emp_id}")
                vacation_priority = st.selectbox("Priorita", ["Nízka", "Stredná", "Vysoká", "Kritická"],
                                                 index=1, key=f"vac_priority_{selected_emp_id}")

                if st.button("Pridať žiadosť", key=f"add_vac_{selected_emp_id}"):
                    if vacation_end >= vacation_start:
                        duration = (vacation_end - vacation_start).days + 1
                        new_request = {
                            "start_date": vacation_start,
                            "end_date": vacation_end,
                            "vacation_type": vacation_type,
                            "reason": vacation_reason,
                            "priority": vacation_priority,
                            "approved": False,
                            "duration": duration,
                            "created_date": date.today()
                        }
                        st.session_state.vacation_requests[selected_emp_id].append(new_request)
                        st.success(f"Žiadosť pridaná! ({duration} dní)")
                        st.rerun()
                    else:
                        st.error("Koniec nemôže byť pred začiatkom!")

        with col2:
            st.write("**Prehľad dovolenky:**")

            # Štatistiky dovolenky
            annual_entitlement = selected_emp.get('annual_vacation', 25)
            approved_annual = sum(
                req["duration"] for req in st.session_state.vacation_requests.get(selected_emp_id, [])
                if req["approved"] and req["vacation_type"] == "Riadna dovolenka"
            )
            pending_annual = sum(
                req["duration"] for req in st.session_state.vacation_requests.get(selected_emp_id, [])
                if not req["approved"] and req["vacation_type"] == "Riadna dovolenka"
            )
            remaining = annual_entitlement - approved_annual

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Ročný nárok", annual_entitlement)
            with col_b:
                st.metric("Využité", approved_annual)
            with col_c:
                st.metric("Zostatok", remaining)

            if pending_annual > 0:
                st.info(f"Čakajúce žiadosti: {pending_annual} dní")

            # Upozornenia
            if remaining < 0:
                st.error("⚠️ Prekročený ročný nárok!")
            elif remaining < 5:
                st.warning("⚠️ Nízky zostatok dovolenky!")

        # Zoznam žiadostí
        st.subheader("📋 Existujúce žiadosti")

        vacation_requests = st.session_state.vacation_requests.get(selected_emp_id, [])
        if vacation_requests:
            for idx, req in enumerate(vacation_requests):
                with st.expander(
                        f"{req['vacation_type']} | {req['start_date']} - {req['end_date']} ({req['duration']} dní)"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Typ:** {req['vacation_type']}")
                        st.write(f"**Obdobie:** {req['start_date']} - {req['end_date']}")
                        st.write(f"**Dĺžka:** {req['duration']} dní")
                        st.write(f"**Priorita:** {req['priority']}")

                    with col2:
                        st.write(f"**Dôvod:** {req['reason']}")
                        st.write(f"**Vytvorené:** {req['created_date']}")
                        status = "✅ Schválené" if req['approved'] else "⏳ Čaká na schválenie"
                        st.write(f"**Status:** {status}")

                    with col3:
                        if not req['approved']:
                            if st.button("✅ Schváliť", key=f"approve_{selected_emp_id}_{idx}"):
                                st.session_state.vacation_requests[selected_emp_id][idx]['approved'] = True
                                st.success("Žiadosť schválená!")
                                st.rerun()
                        else:
                            if st.button("❌ Zrušiť schválenie", key=f"unapprove_{selected_emp_id}_{idx}"):
                                st.session_state.vacation_requests[selected_emp_id][idx]['approved'] = False
                                st.success("Schválenie zrušené!")
                                st.rerun()

                        if st.button("🗑️ Odstrániť", key=f"delete_vac_{selected_emp_id}_{idx}"):
                            st.session_state.vacation_requests[selected_emp_id].pop(idx)
                            st.success("Žiadosť odstránená!")
                            st.rerun()
        else:
            st.info("Žiadne žiadosti o dovolenku")

    # Kalendárny prehľad dovoleniek
    st.subheader("📅 Kalendárny prehľad dovoleniek")

    # Vytvoríme prehľad všetkých dovoleniek
    vacation_overview = []
    for emp_id, requests in st.session_state.vacation_requests.items():
        emp_name = employee_names.get(emp_id, f"EMP {emp_id}")
        for req in requests:
            if req['approved']:
                vacation_overview.append({
                    "Zamestnanec": emp_name,
                    "Typ": req['vacation_type'],
                    "Začiatok": req['start_date'],
                    "Koniec": req['end_date'],
                    "Dni": req['duration'],
                    "Dôvod": req['reason']
                })

    if vacation_overview:
        vacation_df = pd.DataFrame(vacation_overview)
        vacation_df = vacation_df.sort_values('Začiatok')
        st.dataframe(vacation_df, use_container_width=True)

        # Export dovoleniek
        csv_vacation = vacation_df.to_csv(index=False)
        st.download_button(
            "📥 Export dovoleniek (CSV)",
            csv_vacation,
            "dovolenky.csv",
            "text/csv"
        )
    else:
        st.info("Žiadne schválené dovolenky")

with tab5:
    st.subheader("🤝 Spolupráca medzi tímami")

    # Inicializácia session state pre spoluprácu
    if 'collaborations' not in st.session_state:
        st.session_state.collaborations = []

    collaborations = []  # Inicializácia pre tento tab

    if len(teams) < 2:
        st.info("Pre nastavenie spolupráce potrebujete aspoň 2 tímy.")
    else:
        for i, collab_data in enumerate(st.session_state.collaborations):
            with st.expander(f"Spolupráca {i + 1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    team1_options = {t.id: t.name for t in teams}
                    team2_options = {t.id: t.name for t in teams}

                    team1_id = st.selectbox("Prvý tím", list(team1_options.keys()),
                                            format_func=lambda x: team1_options[x], key=f"collab_team1_{i}")
                    team2_id = st.selectbox("Druhý tím", list(team2_options.keys()),
                                            format_func=lambda x: team2_options[x], key=f"collab_team2_{i}")

                with col2:
                    max_shared = st.number_input("Max. zdieľaných zamestnancov", 1, 5, 2, key=f"collab_shared_{i}")
                    priority = st.number_input("Priorita spolupráce", 1, 10, 1, key=f"collab_priority_{i}")

                shift_names = [s.name for s in shift_types]
                allowed_shifts = st.multiselect("Povolené smeny pre spoluprácu", shift_names, shift_names,
                                                key=f"collab_shifts_{i}")

                collaborations.append(Collaboration(
                    team1_id=team1_id, team2_id=team2_id, shift_types=allowed_shifts,
                    max_shared_employees=max_shared, priority=priority
                ))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Pridať spoluprácu"):
                st.session_state.collaborations.append({})
                st.rerun()

        with col2:
            if st.button("🗑️ Odstrániť spoluprácu") and st.session_state.collaborations:
                st.session_state.collaborations.pop()
                st.rerun()

with tab6:
    st.subheader("📊 Generovanie plánu")

    # Nastavenie pokrytia automaticky z tímov
    st.subheader("🎯 Automatické pokrytie z tímových požiadaviek")
    coverage = {}

    for team in teams:
        coverage[team.id] = team.company_requirements.target_coverage_per_shift

    # Zobrazenie súhrnu pokrytia
    coverage_summary = []
    for team in teams:
        for shift_name, count in coverage.get(team.id, {}).items():
            coverage_summary.append({
                "Tím": team.name,
                "Smena": shift_name,
                "Cieľové pokrytie": count,
                "Min. pokrytie": team.company_requirements.min_coverage_per_shift.get(shift_name, count),
                "Max. pokrytie": team.company_requirements.max_coverage_per_shift.get(shift_name, count + 1)
            })

    if coverage_summary:
        coverage_df = pd.DataFrame(coverage_summary)
        st.dataframe(coverage_df, use_container_width=True)

    # Možnosť manuálneho prepísania
    st.subheader("⚙️ Manuálne úpravy pokrytia (voliteľné)")
    manual_override = st.checkbox("Povoliť manuálne úpravy pokrytia")

    if manual_override:
        for team in teams:
            st.write(f"**{team.name}** ({team.id})")
            coverage[team.id] = {}

            cols = st.columns(len(shift_types))
            for i, shift in enumerate(shift_types):
                with cols[i]:
                    original_count = team.company_requirements.target_coverage_per_shift.get(shift.name, 1)
                    count = st.number_input(
                        f"{shift.name}",
                        min_value=0, max_value=10, value=original_count,
                        key=f"manual_coverage_{team.id}_{shift.name}"
                    )
                    coverage[team.id][shift.name] = count

    # Aktualizácia dovoleniek do employee objektov
    for emp in employees:
        emp.vacation_requests = []
        if emp.id in st.session_state.vacation_requests:
            for req_data in st.session_state.vacation_requests[emp.id]:
                if req_data['approved']:
                    vacation_req = VacationRequest(
                        start_date=req_data['start_date'],
                        end_date=req_data['end_date'],
                        vacation_type=VacationType(req_data['vacation_type']),
                        reason=req_data['reason'],
                        approved=True,
                        priority=Priority(req_data['priority'])
                    )
                    emp.vacation_requests.append(vacation_req)

    # Diagnostika
    st.subheader("🔍 Diagnostika plánu")

    total_shifts_needed = 0
    total_employees = len(employees)
    period_days = (end_date - start_date).days + 1

    for team_id, team_coverage in coverage.items():
        for shift_name, count in team_coverage.items():
            total_shifts_needed += count * period_days

    avg_shifts_per_employee = total_shifts_needed / total_employees if total_employees > 0 else 0
    total_hours_needed = sum(
        count * period_days * next(s.duration_hours() for s in shift_types if s.name == shift_name)
        for team_coverage in coverage.values()
        for shift_name, count in team_coverage.items()
    )
    avg_hours_per_employee = total_hours_needed / total_employees if total_employees > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Celkový počet smien", total_shifts_needed)
    with col2:
        st.metric("Počet zamestnancov", total_employees)
    with col3:
        st.metric("Priemerné smeny/zamestnanec", f"{avg_shifts_per_employee:.1f}")
    with col4:
        st.metric("Priemerné hodiny/zamestnanec", f"{avg_hours_per_employee:.1f}")

    # Upozornenia a diagnostika
    unavailable_days = 0
    for emp in employees:
        for req in emp.vacation_requests:
            if req.approved:
                unavailable_days += req.get_duration_days()

    if avg_hours_per_employee > 200:
        st.error("❌ Príliš vysoká záťaž! Zamestnanci budú preťažení.")
    elif avg_hours_per_employee > 180:
        st.warning("⚠️ Vysoká záťaž. Plán môže byť náročný na splnenie.")
    else:
        st.success("✅ Rozumná záťaž. Plán by mal byť dobre optimalizovateľný.")

    if unavailable_days > 0:
        st.info(f"ℹ️ Celkovo {unavailable_days} dní dovolenky/neprítomnosti")

    # Analýza dostupnosti zamestnancov
    st.subheader("👥 Analýza dostupnosti zamestnancov")
    availability_analysis = []

    for emp in employees:
        unavailable_count = sum(req.get_duration_days() for req in emp.vacation_requests if req.approved)
        available_days = period_days - unavailable_count
        availability_pct = (available_days / period_days) * 100

        target_hours = emp.work_requirements.monthly_hours_target
        max_possible_hours = available_days * 8  # Predpokladáme max 8h/deň

        availability_analysis.append({
            "Zamestnanec": emp.name,
            "Tím": next((t.name for t in teams if t.id == emp.team_id), "Neznámy"),
            "Dostupné dni": available_days,
            "Dostupnosť (%)": f"{availability_pct:.1f}%",
            "Cieľové hodiny": target_hours,
            "Max. možné hodiny": max_possible_hours,
            "Realizovateľnosť": "✅ OK" if max_possible_hours >= target_hours else "⚠️ Problém"
        })

    availability_df = pd.DataFrame(availability_analysis)
    st.dataframe(availability_df, use_container_width=True)

    # Generovanie plánu
    st.subheader("🚀 Generovanie plánu")

    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit = st.number_input("Časový limit (sekundy)", 60, 600, 180)
    with col2:
        fallback_enabled = st.checkbox("Povoliť automatické znižovanie požiadaviek pri zlyhaní", True)
    with col3:
        solver_mode = st.selectbox("Režim riešenia", ["Rýchly", "Vyvážený", "Presný"])

    if st.button("🧮 Vygenerovať plán", type="primary"):
        if not employees:
            st.error("❌ Musíte pridať aspoň jedného zamestnanca!")
            st.stop()

        if not teams:
            st.error("❌ Musíte pridať aspoň jeden tím!")
            st.stop()

        if not shift_types:
            st.error("❌ Musíte definovať aspoň jednu smenu!")
            st.stop()

        # Zabezpečíme že collaborations je definované
        if 'collaborations' not in locals():
            collaborations = []

        # Nastavenie optimalizačných cieľov podľa výberu
        if optimization_goal == "Minimalizácia nákladov":
            minimize_cost = True
            balance_workload = False
        elif optimization_goal == "Vyváženie záťaže":
            minimize_cost = False
            balance_workload = True
        else:
            minimize_cost = False
            balance_workload = True

        try:
            with st.spinner("Generujem pokročilý plán..."):
                scheduler = AdvancedScheduler(
                    employees=employees,
                    teams=teams,
                    shift_types=shift_types,
                    period_start=start_date,
                    period_end=end_date,
                    coverage=coverage,
                    collaborations=collaborations,
                    consider_skills=consider_skills,
                    balance_workload=balance_workload,
                    minimize_cost=minimize_cost
                )

                schedule_df = scheduler.solve(limit=time_limit)

        except Exception as exc:
            st.error(f"❌ Chyba pri generovaní plánu: {exc}")

            if fallback_enabled:
                st.info("🔄 Pokúšam sa s redukovanými požiadavkami...")
                try:
                    # Drastické zníženie pokrytia
                    reduced_coverage = {}
                    for team_id, team_coverage in coverage.items():
                        reduced_coverage[team_id] = {}
                        for shift_name, count in team_coverage.items():
                            # Znížime na minimum alebo 1
                            reduced_coverage[team_id][shift_name] = max(1, count // 2)

                    # Fallback scheduler s minimálnymi obmedzeniami
                    scheduler_fallback = AdvancedScheduler(
                        employees=employees,
                        teams=teams,
                        shift_types=shift_types,
                        period_start=start_date,
                        period_end=end_date,
                        coverage=reduced_coverage,
                        collaborations=[],  # Vypnúť collaborations
                        consider_skills=False,  # Vypnúť skill checking
                        balance_workload=False,  # Vypnúť balance
                        minimize_cost=False  # Len základná optimalizácia
                    )

                    schedule_df = scheduler_fallback.solve(limit=time_limit)
                    st.warning("⚠️ Plán bol vygenerovaný s minimálnymi požiadavkami!")

                except Exception as fallback_exc:
                    st.error(f"❌ Ani s minimálnymi požiadavkami sa nepodarilo vygenerovať plán: {fallback_exc}")

                    # Posledná šanca - úplne základný plán
                    st.info("🔄 Pokúšam sa s úplne základným plánom...")
                    try:
                        # Minimálne pokrytie - len 1 osoba na smenu
                        minimal_coverage = {}
                        for team_id in coverage.keys():
                            minimal_coverage[team_id] = {}
                            for shift_name in coverage[team_id].keys():
                                minimal_coverage[team_id][shift_name] = 1

                        scheduler_minimal = AdvancedScheduler(
                            employees=employees[:min(3, len(employees))],  # Max 3 zamestnanci
                            teams=teams[:1],  # Len prvý tím
                            shift_types=shift_types[:2],  # Len prvé 2 smeny
                            period_start=start_date,
                            period_end=min(end_date, start_date + timedelta(days=7)),  # Max týždeň
                            coverage=minimal_coverage,
                            collaborations=[],
                            consider_skills=False,
                            balance_workload=False,
                            minimize_cost=False
                        )

                        schedule_df = scheduler_minimal.solve(limit=60)
                        st.warning("⚠️ Vygenerovaný bol iba ukážkový základný plán!")

                    except Exception as final_exc:
                        st.error(f"❌ Nepodarilo sa vygenerovať ani základný plán: {final_exc}")
                        st.error("💡 Skúste:")
                        st.error("- Pridať viac zamestnancov")
                        st.error("- Znížiť počet požadovaných smien")
                        st.error("- Skrátiť plánovacie obdobie")
                        st.error("- Odstrániť preferencie zamestnancov")
                        st.stop()
            else:
                st.stop()

        st.success("✅ Pokročilý plán úspešne vygenerovaný!")

        # Zobrazenie výsledkov v rozšírených tabs
        tab_schedule, tab_summary, tab_teams, tab_costs, tab_vacations = st.tabs([
            "📋 Plán", "📊 Súhrn zamestnancov", "🏢 Súhrn tímov", "💰 Náklady", "🏖️ Dovolenky"
        ])

        with tab_schedule:
            st.subheader("📋 Rozvrh smien")

            # Kompletný kalendár pre všetkých zamestnancov
            if not schedule_df.empty:
                # Vytvoríme kompletný daterange pre celé obdobie
                full_date_range = pd.date_range(start=start_date, end=end_date)

                # Vytvoríme prázdnu tabuľku pre všetkých zamestnancov a všetky dni
                schedule_matrix = {}

                # Inicializujeme prázdnu maticu pre každého zamestnanca
                for emp in employees:
                    schedule_matrix[emp.name] = {
                        'Tím': next((t.name for t in teams if t.id == emp.team_id), "Neznámy")
                    }
                    # Pridáme každý deň ako stĺpec s defaultnou hodnotou "-"
                    for single_date in full_date_range:
                        day_key = single_date.strftime('%d.%m')
                        schedule_matrix[emp.name][day_key] = '-'

                # Naplníme skutočné smeny z schedule_df
                for _, row in schedule_df.iterrows():
                    emp_name = row['Zamestnanec']
                    date_obj = pd.to_datetime(row['Dátum']).strftime('%d.%m')
                    shift_name = row['Zmena']

                    # Mapovanie názvov smien na krátke kódy
                    shift_mapping = {
                        'Denná': 'D',
                        'Poobedná': 'P',
                        'Nočná': 'N'
                    }

                    # Použijeme mapovanie alebo prvé písmeno ako fallback
                    short_shift = shift_mapping.get(shift_name, shift_name[:1].upper() if shift_name else '-')

                    if emp_name in schedule_matrix:
                        schedule_matrix[emp_name][date_obj] = short_shift

                # Konvertujeme na DataFrame
                display_df = pd.DataFrame.from_dict(schedule_matrix, orient='index')

                # Pridáme súčty hodín a nákladov
                hours_summary = schedule_df.groupby('Zamestnanec')['Hodiny'].sum()
                cost_summary = schedule_df.groupby('Zamestnanec')['Náklady'].sum()

                display_df['Celkom hodín'] = display_df.index.map(hours_summary).fillna(0)
                display_df['Celkom nákladov (€)'] = display_df.index.map(cost_summary).fillna(0).round(2)

                # Pridáme informácie o cieľových hodinách
                target_hours_map = {emp.name: emp.work_requirements.monthly_hours_target for emp in employees}
                display_df['Cieľové hodiny'] = display_df.index.map(target_hours_map).fillna(160)
                display_df['Rozdiel od cieľa'] = display_df['Celkom hodín'] - display_df['Cieľové hodiny']

                # Zoradíme stĺpce - najprv Tím, potom dátumy, potom súčty
                date_columns = [col for col in display_df.columns if
                                '.' in col and 'Celkom' not in col and 'Cieľové' not in col and 'Rozdiel' not in col]
                date_columns_sorted = sorted(date_columns, key=lambda x: datetime.strptime(x + '.2025', '%d.%m.%Y'))

                column_order = ['Tím'] + date_columns_sorted + ['Celkom hodín', 'Cieľové hodiny', 'Rozdiel od cieľa',
                                                                'Celkom nákladov (€)']
                display_df = display_df[column_order]

                # Resetujeme index aby sa zamestnanec zobrazil ako stĺpec
                display_df.reset_index(inplace=True)
                display_df.rename(columns={'index': 'Zamestnanec'}, inplace=True)

                st.dataframe(display_df, use_container_width=True, height=500)

                # Legenda pre krátke názvy
                st.subheader("🔤 Legenda smien")

                # Vytvoríme mapovanie pre legendu
                shift_mapping = {
                    'Denná': 'D',
                    'Poobedná': 'P',
                    'Nočná': 'N'
                }

                legend_cols = st.columns(len(shift_types) + 1)  # +1 pre "-"
                for i, shift in enumerate(shift_types):
                    with legend_cols[i]:
                        short = shift_mapping.get(shift.name, shift.name[:1].upper())
                        hours = shift.duration_hours()
                        st.write(f"**{short}** = {shift.name} ({hours}h)")

                # Pridáme legendu pre voľný deň
                with legend_cols[-1]:
                    st.write(f"**-** = Voľný deň")

                # Export možnosti
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 Stiahnuť plán (CSV)",
                        csv_data,
                        "plan_smien_kompletny.csv",
                        "text/csv"
                    )

                with col2:
                    # Excel export s formátovaním
                    excel_csv = display_df.to_csv(index=False)
                    st.download_button(
                        "📊 Stiahnuť pre Excel",
                        excel_csv,
                        "plan_smien.csv",
                        "text/csv"
                    )

                with col3:
                    json_data = schedule_df.to_json(orient="records", date_format="iso")
                    st.download_button(
                        "📥 Stiahnuť detaily (JSON)",
                        json_data,
                        "plan_detaily.json",
                        "application/json"
                    )

                # Prehľad pokrytia po dňoch
                st.subheader("📊 Prehľad pokrytia")

                coverage_analysis = schedule_df.groupby(['Dátum', 'Zmena']).size().reset_index(name='Počet')
                coverage_pivot = coverage_analysis.pivot(index='Dátum', columns='Zmena', values='Počet').fillna(0)

                st.dataframe(coverage_pivot, use_container_width=True)

                # Kontrola nedostatočného pokrytia
                issues = []
                for team in teams:
                    for shift_name, required_count in coverage.get(team.id, {}).items():
                        if required_count > 0:
                            for single_date in full_date_range:
                                actual_count = len(schedule_df[
                                                       (pd.to_datetime(
                                                           schedule_df['Dátum']).dt.date == single_date.date()) &
                                                       (schedule_df['Zmena'] == shift_name) &
                                                       (schedule_df['Tím'] == team.name)
                                                       ])
                                if actual_count < required_count:
                                    issues.append({
                                        'Dátum': single_date.date(),
                                        'Tím': team.name,
                                        'Zmena': shift_name,
                                        'Požadované': required_count,
                                        'Skutočné': actual_count,
                                        'Chýba': required_count - actual_count
                                    })

                if issues:
                    st.warning("⚠️ Nájdené problémy s pokrytím:")
                    issues_df = pd.DataFrame(issues)
                    st.dataframe(issues_df, use_container_width=True)
                else:
                    st.success("✅ Všetky požiadavky na pokrytie sú splnené!")

            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_summary:
            st.subheader("📊 Rozšírený súhrn zamestnancov")
            summaries = scheduler.get_summary(schedule_df)

            if not summaries["employees"].empty:
                st.dataframe(summaries["employees"], use_container_width=True)

                # Grafy výkonnosti
                fig_data = summaries["employees"].copy()
                if "Odpracované hodiny" in fig_data.columns:
                    st.subheader("📈 Porovnanie odpracovaných vs cieľových hodín")
                    chart_data = fig_data[['Zamestnanec', 'Odpracované hodiny', 'Cieľové hodiny']].set_index(
                        'Zamestnanec')
                    st.bar_chart(chart_data)
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_teams:
            st.subheader("🏢 Rozšírený súhrn tímov")

            if not summaries["teams"].empty:
                st.dataframe(summaries["teams"], use_container_width=True)

                # Pie chart pre rozdelenie nákladov medzi tímami
                if "Celkom nákladov" in summaries["teams"].columns:
                    st.subheader("💰 Rozdelenie nákladov medzi tímami")
                    fig_data = summaries["teams"].set_index("Tím")["Celkom nákladov"]
                    st.bar_chart(fig_data)
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_costs:
            st.subheader("💰 Detailná analýza nákladov")

            if not schedule_df.empty:
                # Celkové náklady
                total_cost = schedule_df["Náklady"].sum()
                total_hours = schedule_df["Hodiny"].sum()
                avg_hourly = total_cost / total_hours if total_hours > 0 else 0
                weekend_cost = schedule_df[schedule_df["Je_víkend"] == True][
                    "Náklady"].sum() if "Je_víkend" in schedule_df.columns else 0
                regular_cost = total_cost - weekend_cost

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Celkové náklady", f"{total_cost:.2f} €")
                with col2:
                    st.metric("Víkendové náklady", f"{weekend_cost:.2f} €")
                with col3:
                    st.metric("Celkové hodiny", f"{total_hours:.1f}")
                with col4:
                    st.metric("Priemerná sadzba", f"{avg_hourly:.2f} €/h")

                # Náklady podľa tímov
                team_costs = schedule_df.groupby("Tím").agg({
                    "Náklady": "sum",
                    "Hodiny": "sum"
                }).reset_index()
                team_costs["Priemerná sadzba"] = team_costs["Náklady"] / team_costs["Hodiny"]

                st.subheader("Náklady podľa tímov")
                st.dataframe(team_costs, use_container_width=True)

                # Náklady podľa smien
                shift_costs = schedule_df.groupby("Zmena").agg({
                    "Náklady": ["sum", "mean"],
                    "Hodiny": "sum"
                }).reset_index()
                shift_costs.columns = ["Zmena", "Celkové náklady", "Priemerné náklady", "Celkové hodiny"]

                st.subheader("Náklady podľa smien")
                st.dataframe(shift_costs, use_container_width=True)

                # Graf nákladov
                st.subheader("📊 Vizualizácia nákladov")
                st.bar_chart(team_costs.set_index("Tím")["Náklady"])
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_vacations:
            st.subheader("🏖️ Analýza dovoleniek a súlad s plánom")

            if not summaries["vacations"].empty:
                st.dataframe(summaries["vacations"], use_container_width=True)

                # Analýza vplyvu dovoleniek na plán
                st.subheader("📊 Vplyv dovoleniek na pracovný plán")

                vacation_impact = []
                for emp in employees:
                    emp_schedule = schedule_df[
                        schedule_df["Zamestnanec"] == emp.name] if not schedule_df.empty else pd.DataFrame()
                    worked_days = len(emp_schedule)
                    total_days = period_days
                    vacation_days = sum(req.get_duration_days() for req in emp.vacation_requests if req.approved)
                    available_days = total_days - vacation_days

                    if available_days > 0:
                        utilization = (worked_days / available_days) * 100
                    else:
                        utilization = 0

                    vacation_impact.append({
                        "Zamestnanec": emp.name,
                        "Celkové dni": total_days,
                        "Dovolenkové dni": vacation_days,
                        "Dostupné dni": available_days,
                        "Odpracované dni": worked_days,
                        "Využitie (%)": f"{utilization:.1f}%",
                        "Status": "✅ Optimálne" if 70 <= utilization <= 90 else "⚠️ Kontrola potrebná"
                    })

                vacation_impact_df = pd.DataFrame(vacation_impact)
                st.dataframe(vacation_impact_df, use_container_width=True)
            else:
                st.info("Žiadne dáta o dovolenkách na zobrazenie.")

with tab7:
    st.subheader("📈 Pokročilé analýzy a reporty")

    if 'schedule_df' in locals() and not schedule_df.empty:

        # KPI Dashboard
        st.subheader("🎯 KPI Dashboard")

        # Výpočet KPI
        total_employees = len(employees)
        total_scheduled_hours = schedule_df["Hodiny"].sum()
        total_target_hours = sum(emp.work_requirements.monthly_hours_target for emp in employees)
        target_achievement = (total_scheduled_hours / total_target_hours * 100) if total_target_hours > 0 else 0

        unique_employees_scheduled = schedule_df["Zamestnanec"].nunique()
        employee_utilization = (unique_employees_scheduled / total_employees * 100) if total_employees > 0 else 0

        avg_satisfaction = 85.0  # Simulované - v reálnej aplikácii by sa počítalo z preferencií

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Splnenie cieľových hodín",
                f"{target_achievement:.1f}%",
                delta=f"{target_achievement - 100:.1f}%" if target_achievement != 0 else None
            )

        with col2:
            st.metric(
                "Využitie zamestnancov",
                f"{employee_utilization:.1f}%",
                delta=f"{employee_utilization - 100:.1f}%" if employee_utilization != 0 else None
            )

        with col3:
            total_cost = schedule_df["Náklady"].sum()
            budget_limit = 50000  # Simulovaný rozpočet
            budget_usage = (total_cost / budget_limit * 100) if budget_limit > 0 else 0
            st.metric(
                "Využitie rozpočtu",
                f"{budget_usage:.1f}%",
                delta=f"{budget_usage - 100:.1f}%" if budget_usage != 0 else None
            )

        with col4:
            st.metric("Spokojnosť zamestnancov", f"{avg_satisfaction:.1f}%")

        # Trendy a analýzy
        st.subheader("📊 Týždenné trendy")

        # Analýza po týždňoch
        schedule_with_week = schedule_df.copy()
        schedule_with_week['Dátum'] = pd.to_datetime(schedule_with_week['Dátum'])
        schedule_with_week['Týždeň'] = schedule_with_week['Dátum'].dt.isocalendar().week

        weekly_analysis = schedule_with_week.groupby('Týždeň').agg({
            'Hodiny': 'sum',
            'Náklady': 'sum',
            'Zamestnanec': 'nunique'
        }).reset_index()
        weekly_analysis.columns = ['Týždeň', 'Celkové hodiny', 'Celkové náklady', 'Aktívni zamestnanci']

        st.dataframe(weekly_analysis, use_container_width=True)

        # Graf týždenných trendov
        st.line_chart(weekly_analysis.set_index('Týždeň')[['Celkové hodiny', 'Aktívni zamestnanci']])

        # Analýza výkonnosti tímov
        st.subheader("🏢 Porovnanie výkonnosti tímov")

        team_performance = schedule_df.groupby('Tím').agg({
            'Hodiny': ['sum', 'mean'],
            'Náklady': ['sum', 'mean'],
            'Zamestnanec': 'nunique'
        }).reset_index()

        team_performance.columns = [
            'Tím', 'Celkové hodiny', 'Priemerné hodiny/smena',
            'Celkové náklady', 'Priemerné náklady/smena', 'Počet zamestnancov'
        ]

        # Výpočet efektivity (hodiny na zamestnanca)
        team_performance['Efektivita (h/zamestnanec)'] = team_performance['Celkové hodiny'] / team_performance[
            'Počet zamestnancov']

        st.dataframe(team_performance, use_container_width=True)

        # Identifikácia problémov a odporúčania
        st.subheader("⚠️ Identifikované problémy a odporúčania")

        problems = []
        recommendations = []

        # Kontrola preťažených zamestnancov
        employee_hours = schedule_df.groupby('Zamestnanec')['Hodiny'].sum()
        overworked = employee_hours[employee_hours > 200]
        if not overworked.empty:
            problems.append(f"Preťažení zamestnanci: {', '.join(overworked.index)}")
            recommendations.append("Prerozdeliť záťaž alebo pridať nových zamestnancov")

        # Kontrola nedostatočne využitých zamestnancov
        underutilized = employee_hours[employee_hours < 120]
        if not underutilized.empty:
            problems.append(f"Nedostatočne využití zamestnanci: {', '.join(underutilized.index)}")
            recommendations.append("Zvýšiť záťaž alebo prehodnotiť potrebu pozícií")

        # Kontrola vysokých nákladov
        if total_cost > budget_limit:
            problems.append(f"Prekročený rozpočet o {total_cost - budget_limit:.2f} €")
            recommendations.append("Optimalizovať rozloženie smien alebo znížiť prémiové hodiny")

        # Kontrola pokrytia víkendov
        weekend_coverage = schedule_df[schedule_df.get('Je_víkend', False) == True]
        if weekend_coverage.empty and any(shift.is_weekend_applicable for shift in shift_types):
            problems.append("Nedostatočné pokrytie víkendov")
            recommendations.append("Pridať víkendové smeny alebo motivovať zamestnancov prémií")

        if problems:
            st.warning("Identifikované problémy:")
            for i, problem in enumerate(problems, 1):
                st.write(f"{i}. {problem}")

            st.info("Odporúčania:")
            for i, recommendation in enumerate(recommendations, 1):
                st.write(f"{i}. {recommendation}")
        else:
            st.success("✅ Žiadne významné problémy neboli identifikované!")

        # Export pokročilých reportov
        st.subheader("📋 Export reportov")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Manažérsky report
            manager_report = {
                "KPI": {
                    "Splnenie_cielovych_hodin": f"{target_achievement:.1f}%",
                    "Vyuzitie_zamestnancov": f"{employee_utilization:.1f}%",
                    "Vyuzitie_rozpoctu": f"{budget_usage:.1f}%",
                    "Celkove_naklady": f"{total_cost:.2f} €"
                },
                "Problemy": problems,
                "Odporucania": recommendations
            }

            manager_json = json.dumps(manager_report, indent=2, ensure_ascii=False)
            st.download_button(
                "📊 Manažérsky report (JSON)",
                manager_json,
                "manager_report.json",
                "application/json"
            )

        with col2:
            # HR report
            hr_data = []
            for emp in employees:
                emp_schedule = schedule_df[schedule_df["Zamestnanec"] == emp.name]
                hr_data.append({
                    "Zamestnanec": emp.name,
                    "ID": emp.id,
                    "Tim": next((t.name for t in teams if t.id == emp.team_id), "Neznámy"),
                    "Odpracovane_hodiny": emp_schedule["Hodiny"].sum(),
                    "Cielove_hodiny": emp.work_requirements.monthly_hours_target,
                    "Naklady": emp_schedule["Náklady"].sum(),
                    "Pocet_smien": len(emp_schedule),
                    "Zostatok_dovolenky": emp.vacation_remaining()
                })

            hr_df = pd.DataFrame(hr_data)
            hr_csv = hr_df.to_csv(index=False)
            st.download_button(
                "👥 HR report (CSV)",
                hr_csv,
                "hr_report.csv",
                "text/csv"
            )

        with col3:
            # Finančný report
            financial_df = team_performance.copy()
            financial_csv = financial_df.to_csv(index=False)
            st.download_button(
                "💰 Finančný report (CSV)",
                financial_csv,
                "financial_report.csv",
                "text/csv"
            )

    else:
        st.info("Pre zobrazenie analýz najprv vygenerujte plán v záložke 'Generovanie'")

# Footer
st.markdown("---")
st.markdown("**PlanME Pro** - Enterprise Team Scheduler | Verzia 3.0")
st.markdown("💡 *Pokročilý plánovač s podporou dovoleniek, firemných požiadaviek a detailných analýz*")

# Sidebar s dodatočnými informáciami
with st.sidebar:
    st.markdown("---")
    st.subheader("📋 Rýchly prehľad")

    if 'employees' in locals():
        st.write(f"👥 Zamestnanci: {len(employees)}")
    if 'teams' in locals():
        st.write(f"🏢 Tímy: {len(teams)}")
    if 'shift_types' in locals():
        st.write(f"⏰ Smeny: {len(shift_types)}")

    # Počet dní plánovania
    if 'start_date' in locals() and 'end_date' in locals():
        total_period_days = (end_date - start_date).days + 1
        st.write(f"📅 Obdobie: {total_period_days} dní")

    # Celkový počet dovolenkových žiadostí
    if 'st' in locals() and hasattr(st, 'session_state') and 'vacation_requests' in st.session_state:
        total_vacation_requests = sum(len(requests) for requests in st.session_state.vacation_requests.values())
        st.write(f"🏖️ Dovolenkové žiadosti: {total_vacation_requests}")

    st.markdown("---")
    st.subheader("💡 Tipy na optimalizáciu")
    st.markdown("""
    - Udržujte vyváženú záťaž medzi zamestnancami
    - Používajte preferencie smien pre vyššiu spokojnosť
    - Pravidelne kontrolujte využitie rozpočtu
    - Plánujte dovolenky s dostatočným predstihom
    - Monitorujte víkendové pokrytie
    """)

    st.markdown("---")
    st.caption("© 2025 PlanME Pro - All rights reserved")# rn.py - Hlavná aplikácia s databázovou integráciou

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Sequence, Tuple, Optional, Set
from enum import Enum
import json
import calendar

import pandas as pd
from dateutil.rrule import rrule, DAILY
from ortools.sat.python import cp_model
import streamlit as st

try:
    from streamlit_calendar import date_picker  # type: ignore
except ImportError:
    date_picker = None

# Import databázového modulu
from database_manager import (
    DatabaseManager, init_database, sync_to_database,
    load_from_database, add_database_controls,
    save_generated_schedule, load_existing_schedule
)


class SkillLevel(Enum):
    BEGINNER = "Zaciatocník"
    INTERMEDIATE = "Pokrocilý"
    ADVANCED = "Expert"
    SUPERVISOR = "Supervízor"


class ContractType(Enum):
    FULL_TIME = "Plný úvazok"
    PART_TIME = "Ciastočný úvazok"
    TEMPORARY = "Docasný"
    CONTRACT = "Zmluvný"


class VacationType(Enum):
    ANNUAL = "Riadna dovolenka"
    SICK = "Nemocenská"
    PERSONAL = "Osobné volno"
    MATERNITY = "Materská/otcovská"
    UNPAID = "Neplatené voľno"
    TRAINING = "Skolenie"
    COMPENSATION = "Náhradné volno"


class Priority(Enum):
    LOW = "Nízka"
    MEDIUM = "Stredná"
    HIGH = "Vysoká"
    CRITICAL = "Kritická"


@dataclass
class Skill:
    name: str
    level: SkillLevel
    priority: int = 1  # 1 = najvyššia, 5 = najnižšia
    certification_expiry: Optional[date] = None


@dataclass
class VacationRequest:
    start_date: date
    end_date: date
    vacation_type: VacationType = VacationType.ANNUAL
    reason: str = ""
    approved: bool = False
    priority: Priority = Priority.MEDIUM
    created_date: date = field(default_factory=date.today)

    def get_duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def overlaps_with(self, other_start: date, other_end: date) -> bool:
        return not (self.end_date < other_start or self.start_date > other_end)


@dataclass
class ShiftType:
    name: str
    start: time
    end: time
    min_rest_hours_after: int = 11
    rest_days_after: int = 0
    required_skills: List[str] = field(default_factory=list)
    min_skill_level: SkillLevel = SkillLevel.BEGINNER
    difficulty_multiplier: float = 1.0
    premium_pay: float = 0.0
    is_weekend_applicable: bool = True
    max_consecutive_days: int = 7
    min_employees: int = 1
    max_employees: int = 10

    def duration_hours(self) -> float:
        start_dt = datetime.combine(date.today(), self.start)
        end_dt = datetime.combine(date.today(), self.end)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        return (end_dt - start_dt).total_seconds() / 3600


@dataclass
class WorkRequirement:
    monthly_hours_target: int = 160
    weekly_hours_min: int = 20
    weekly_hours_max: int = 48
    max_overtime_hours: int = 10
    min_days_off_per_week: int = 1
    max_consecutive_work_days: int = 6


@dataclass
class Employee:
    id: str
    name: str
    team_id: str
    max_consecutive_days: int = 5
    preferences: List[str] = field(default_factory=list)
    annual_vacation_days: int = 25
    max_night_shifts: int = 999
    contract_type: ContractType = ContractType.FULL_TIME
    skills: List[Skill] = field(default_factory=list)
    hourly_rate: float = 15.0
    seniority_years: int = 0
    can_work_alone: bool = True
    needs_supervision: bool = False
    email: str = ""
    phone: str = ""
    emergency_contact: str = ""
    notes: str = ""

    # Nové rozšírené atribúty
    vacation_requests: List[VacationRequest] = field(default_factory=list)
    work_requirements: WorkRequirement = field(default_factory=WorkRequirement)
    overtime_eligible: bool = True
    weekend_work_allowed: bool = True
    night_shift_restriction: bool = False
    start_date: Optional[date] = None
    probation_end_date: Optional[date] = None
    performance_rating: float = 3.0  # 1-5 škála
    languages: List[str] = field(default_factory=list)

    def is_available(self, d: date) -> bool:
        # Kontrola dovolenkových žiadostí
        for vacation in self.vacation_requests:
            if vacation.approved and vacation.start_date <= d <= vacation.end_date:
                return False
        return True

    def vacation_remaining(self) -> int:
        approved_annual = sum(
            vr.get_duration_days() for vr in self.vacation_requests
            if vr.approved and vr.vacation_type == VacationType.ANNUAL
        )
        return self.annual_vacation_days - approved_annual

    def has_skill(self, skill_name: str, min_level: SkillLevel = SkillLevel.BEGINNER) -> bool:
        for skill in self.skills:
            if skill.name == skill_name:
                skill_levels = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED,
                                SkillLevel.SUPERVISOR]
                return skill_levels.index(skill.level) >= skill_levels.index(min_level)
        return False

    def get_skill_level(self, skill_name: str) -> Optional[SkillLevel]:
        for skill in self.skills:
            if skill.name == skill_name:
                return skill.level
        return None

    def get_pending_vacation_days(self) -> int:
        return sum(
            vr.get_duration_days() for vr in self.vacation_requests
            if not vr.approved and vr.vacation_type == VacationType.ANNUAL
        )


@dataclass
class CompanyRequirements:
    min_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: min_people}
    max_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: max_people}
    target_coverage_per_shift: Dict[str, int] = field(default_factory=dict)  # {shift_name: target_people}
    weekend_multiplier: float = 1.0  # Násobiteľ pokrytia pre víkendy
    holiday_multiplier: float = 0.5  # Násobiteľ pokrytia pre sviatky
    emergency_contact_required: bool = True
    supervisor_always_present: bool = False


@dataclass
class Team:
    id: str
    name: str
    description: str = ""
    manager_id: Optional[str] = None
    priority: int = 1
    budget_limit: Optional[float] = None
    can_collaborate_with: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    color: str = "#4CAF50"

    # Nové pokročilé atribúty
    company_requirements: CompanyRequirements = field(default_factory=CompanyRequirements)
    department: str = ""
    cost_center: str = ""
    location: str = ""

    def get_employees(self, all_employees: List[Employee]) -> List[Employee]:
        return [emp for emp in all_employees if emp.team_id == self.id]


@dataclass
class Collaboration:
    team1_id: str
    team2_id: str
    shift_types: List[str] = field(default_factory=list)
    max_shared_employees: int = 2
    priority: int = 1


class AdvancedScheduler:
    def __init__(
            self,
            employees: Sequence[Employee],
            teams: Sequence[Team],
            shift_types: Sequence[ShiftType],
            period_start: date,
            period_end: date,
            coverage: Dict[str, Dict[str, int]] | None = None,
            collaborations: List[Collaboration] = None,
            max_total_hours_per_employee: int | None = None,
            consider_skills: bool = True,
            balance_workload: bool = True,
            minimize_cost: bool = False,
            company_requirements: CompanyRequirements = None
    ) -> None:
        self.employees = list(employees)
        self.teams = list(teams)
        self.shift_types = list(shift_types)
        self.period_start = period_start
        self.period_end = period_end
        self.coverage = coverage or {}
        self.collaborations = collaborations or []
        self.max_total_hours_per_employee = max_total_hours_per_employee
        self.consider_skills = consider_skills
        self.balance_workload = balance_workload
        self.minimize_cost = minimize_cost
        self.company_requirements = company_requirements or CompanyRequirements()

        self._dates = list(rrule(DAILY, dtstart=period_start, until=period_end))
        self.model = cp_model.CpModel()
        self._build_vars()
        self._add_constraints()
        self._set_objective()

    def _build_vars(self) -> None:
        self.x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
        for e in range(len(self.employees)):
            for d in range(len(self._dates)):
                for s in range(len(self.shift_types)):
                    self.x[(e, d, s)] = self.model.NewBoolVar(f"x_{e}_{d}_{s}")

    def _is_weekend(self, date_obj: date) -> bool:
        return date_obj.weekday() >= 5  # Sobota = 5, Nedeľa = 6

    def _add_constraints(self) -> None:
        ne, nd, ns = len(self.employees), len(self._dates), len(self.shift_types)

        # ZÁKLADNÉ POKRYTIE - ZJEDNODUŠENÉ
        for d in range(nd):
            for s_idx, shift in enumerate(self.shift_types):
                # Jednoduché pokrytie - aspoň minimálne požiadavky
                total_required = 0
                for team in self.teams:
                    base_required = self.coverage.get(team.id, {}).get(shift.name, 0)
                    total_required += base_required

                if total_required > 0:
                    assigned_to_shift = sum(self.x[(e, d, s_idx)] for e in range(ne))
                    # Iba minimálne pokrytie, bez maxím
                    self.model.Add(assigned_to_shift >= max(1, total_required))

        # Každý zamestnanec max. jedna smena za deň
        for e in range(ne):
            for d in range(nd):
                self.model.Add(sum(self.x[(e, d, s)] for s in range(ns)) <= 1)

        # ZJEDNODUŠENÉ PRACOVNÉ POŽIADAVKY
        for e_idx, emp in enumerate(self.employees):
            # Dostupnosť (dovolenky) - JEDINÉ TVRDÉ OBMEDZENIE
            for d_idx, dt in enumerate(self._dates):
                if not emp.is_available(dt.date()):
                    for s in range(ns):
                        self.model.Add(self.x[(e_idx, d_idx, s)] == 0)

            # Preferencie - IBA AK SÚ DEFINOVANÉ
            if emp.preferences:
                allowed_shifts = {
                    s for s in range(ns) if self.shift_types[s].name in emp.preferences
                }
                if allowed_shifts and len(allowed_shifts) < ns:  # Iba ak nie sú všetky smeny povolené
                    for d_idx, dt in enumerate(self._dates):
                        if emp.is_available(dt.date()):
                            for s in range(ns):
                                if s not in allowed_shifts:
                                    self.model.Add(self.x[(e_idx, d_idx, s)] == 0)

            # MÄKKÉ LIMITY - iba ak je balance_workload True
            if self.balance_workload:
                # Maximálne po sebe idúce dni - VEĽMI JEDNODUCHÉ
                max_consecutive = min(emp.work_requirements.max_consecutive_work_days, 7)
                if max_consecutive < 7:  # Iba ak je nastavené
                    for start in range(nd - max_consecutive):
                        if start + max_consecutive < nd:
                            consecutive_work = sum(
                                sum(self.x[(e_idx, start + i, s)] for s in range(ns))
                                for i in range(max_consecutive + 1)
                            )
                            self.model.Add(consecutive_work <= max_consecutive)

                # Týždenné limity - VEĽMI VOĽNÉ
                weeks = math.ceil(nd / 7)
                for week in range(weeks):
                    week_start = week * 7
                    week_end = min(week_start + 7, nd)

                    work_days = sum(
                        sum(self.x[(e_idx, d, s)] for s in range(ns))
                        for d in range(week_start, week_end)
                    )

                    # Maximálne 6 dní v týždni
                    max_work_days = min(week_end - week_start, 6)
                    self.model.Add(work_days <= max_work_days)

        # Skillové požiadavky - IBA AK SÚ KRITICKÉ
        if self.consider_skills:
            for s_idx, shift in enumerate(self.shift_types):
                if shift.required_skills:
                    for d in range(nd):
                        for e_idx, emp in enumerate(self.employees):
                            # Iba ak zamestnancovi úplne chýbajú potrebné skills
                            missing_critical_skills = [
                                skill for skill in shift.required_skills
                                if not emp.has_skill(skill, SkillLevel.BEGINNER)
                            ]
                            if missing_critical_skills:
                                self.model.Add(self.x[(e_idx, d, s_idx)] == 0)

        # Odpočinok po nočných smenách - IBA PRE NOČNÉ
        for s_idx, shift in enumerate(self.shift_types):
            if shift.rest_days_after > 0 and "nočná" in shift.name.lower():
                for e_idx in range(ne):
                    for d_idx in range(nd - 1):  # Iba jeden deň odpočinku
                        if d_idx + 1 < nd:
                            for s2 in range(ns):
                                self.model.Add(
                                    self.x[(e_idx, d_idx, s_idx)] + self.x[(e_idx, d_idx + 1, s2)] <= 1
                                )

        # Nočné smeny - IBA ÚPLNÝ ZÁKAZ
        for s_idx, shift in enumerate(self.shift_types):
            if "nočná" in shift.name.lower():
                for e_idx, emp in enumerate(self.employees):
                    if emp.night_shift_restriction:
                        for d in range(nd):
                            self.model.Add(self.x[(e_idx, d, s_idx)] == 0)

    def _set_objective(self) -> None:
        ne, nd, ns = len(self.employees), len(self._dates), len(self.shift_types)

        if self.minimize_cost:
            # Jednoduchá minimalizácia nákladov
            total_cost = []
            for e_idx, emp in enumerate(self.employees):
                for d in range(nd):
                    for s_idx, shift in enumerate(self.shift_types):
                        base_cost = int(emp.hourly_rate * shift.duration_hours() * 100)
                        total_cost.append(self.x[(e_idx, d, s_idx)] * base_cost)
            self.model.Minimize(sum(total_cost))

        elif self.balance_workload:
            # Jednoduché vyváženie záťaže
            max_shifts = self.model.NewIntVar(0, nd, "max_shifts")
            min_shifts = self.model.NewIntVar(0, nd, "min_shifts")

            for e in range(ne):
                total_shifts = sum(self.x[(e, d, s)] for d in range(nd) for s in range(ns))
                self.model.Add(total_shifts <= max_shifts)
                self.model.Add(total_shifts >= min_shifts)

            self.model.Minimize(max_shifts - min_shifts)

        else:
            # Maximalizovať celkové priradenie smien (jednoduchý fallback)
            total_assignments = sum(
                self.x[(e, d, s)] for e in range(ne) for d in range(nd) for s in range(ns)
            )
            self.model.Maximize(total_assignments)

    def solve(self, limit: int = 180) -> pd.DataFrame:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        status = solver.Solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                "Riešenie sa nenašlo v čase limitu. Skúste znížiť požiadavky alebo pridať viac zamestnancov.")

        rows: List[Dict[str, object]] = []
        for d_idx, dt_rule in enumerate(self._dates):
            dt = dt_rule.date()
            for s_idx, stype in enumerate(self.shift_types):
                for e_idx, emp in enumerate(self.employees):
                    if solver.BooleanValue(self.x[(e_idx, d_idx, s_idx)]):
                        team = next((t for t in self.teams if t.id == emp.team_id), None)

                        # Výpočet nákladov s bonusmi
                        is_weekend = self._is_weekend(dt)
                        base_cost = emp.hourly_rate * stype.duration_hours()
                        weekend_bonus = 1.2 if is_weekend else 1.0
                        shift_premium = 1 + stype.premium_pay
                        final_cost = base_cost * weekend_bonus * shift_premium

                        rows.append({
                            "Dátum": dt,
                            "Zmena": stype.name,
                            "Zamestnanec": emp.name,
                            "ID": emp.id,
                            "Tím": team.name if team else "Neznámy",
                            "Hodiny": stype.duration_hours(),
                            "Náklady": final_cost,
                            "Je_víkend": is_weekend,
                            "Víkendový_bonus": weekend_bonus,
                            "Prémia_smeny": shift_premium
                        })
        return pd.DataFrame(rows)

    def get_summary(self, schedule: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if not schedule.empty:
            print(f"Dostupné stĺpce v schedule_df: {list(schedule.columns)}")
            print(f"Počet riadkov: {len(schedule)}")

        # Rozšírený súhrn pre zamestnancov
        employee_summary = []
        for emp in self.employees:
            if "Zamestnanec" in schedule.columns:
                emp_data = schedule[schedule["Zamestnanec"] == emp.name]
            else:
                emp_data = schedule.iloc[0:0]

            total_hours = emp_data["Hodiny"].sum() if "Hodiny" in emp_data.columns else 0
            total_cost = emp_data["Náklady"].sum() if "Náklady" in emp_data.columns else 0
            weekend_hours = emp_data[emp_data["Je_víkend"] == True][
                "Hodiny"].sum() if "Je_víkend" in emp_data.columns else 0
            shift_counts = emp_data[
                "Zmena"].value_counts().to_dict() if "Zmena" in emp_data.columns and not emp_data.empty else {}

            # Cieľové vs skutočné hodiny
            target_hours = emp.work_requirements.monthly_hours_target
            hours_diff = total_hours - target_hours
            hours_status = "✅ V cieli" if abs(hours_diff) <= 20 else "⚠️ Mimo cieľa"

            employee_summary.append({
                "ID": emp.id,
                "Zamestnanec": emp.name,
                "Tím": next((t.name for t in self.teams if t.id == emp.team_id), "Neznámy"),
                "Odpracované hodiny": total_hours,
                "Cieľové hodiny": target_hours,
                "Rozdiel": hours_diff,
                "Status": hours_status,
                "Víkendové hodiny": weekend_hours,
                "Celkom nákladov": total_cost,
                "Zostatok dovolenky": emp.vacation_remaining(),
                "Čakajúce žiadosti": emp.get_pending_vacation_days(),
                **{f"Smeny {k}": v for k, v in shift_counts.items()}
            })

        # Súhrn pre tímy
        team_summary = []
        for team in self.teams:
            if "Tím" in schedule.columns:
                team_data = schedule[schedule["Tím"] == team.name]
            else:
                team_data = schedule.iloc[0:0]

            total_hours = team_data["Hodiny"].sum() if "Hodiny" in team_data.columns else 0
            total_cost = team_data["Náklady"].sum() if "Náklady" in team_data.columns else 0
            employee_count = len(
                team_data["Zamestnanec"].unique()) if "Zamestnanec" in team_data.columns and not team_data.empty else 0
            weekend_cost = team_data[team_data["Je_víkend"] == True][
                "Náklady"].sum() if "Je_víkend" in team_data.columns else 0

            team_summary.append({
                "Tím": team.name,
                "Zamestnanci": employee_count,
                "Celkom hodín": total_hours,
                "Priemerné hodiny/zamestnanec": total_hours / employee_count if employee_count > 0 else 0,
                "Celkom nákladov": total_cost,
                "Víkendové náklady": weekend_cost,
                "Priemerné náklady/zamestnanec": total_cost / employee_count if employee_count > 0 else 0
            })

        # Analýza dovoleniek
        vacation_summary = []
        for emp in self.employees:
            pending_requests = [vr for vr in emp.vacation_requests if not vr.approved]
            approved_requests = [vr for vr in emp.vacation_requests if vr.approved]

            vacation_summary.append({
                "Zamestnanec": emp.name,
                "Ročný nárok": emp.annual_vacation_days,
                "Využité dni": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.ANNUAL),
                "Zostatok": emp.vacation_remaining(),
                "Čakajúce žiadosti": len(pending_requests),
                "Čakajúce dni": sum(
                    vr.get_duration_days() for vr in pending_requests if vr.vacation_type == VacationType.ANNUAL),
                "Nemocenské dni": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.SICK),
                "Osobné voľno": sum(
                    vr.get_duration_days() for vr in approved_requests if vr.vacation_type == VacationType.PERSONAL)
            })

        return {
            "employees": pd.DataFrame(employee_summary),
            "teams": pd.DataFrame(team_summary),
            "vacations": pd.DataFrame(vacation_summary)
        }


# Inicializácia databázy pri prvom spustení
if 'db_initialized' not in st.session_state:
    if init_database():
        st.session_state.db_initialized = True
    else:
        st.error("❌ Chyba pri inicializácii databázy")

# Streamlit UI - Pokročilá verzia s databázou
st.set_page_config(page_title="PlanME Pro – Enterprise Scheduler", page_icon="🏢", layout="wide")
st.title("🏢 PlanME Pro – Enterprise Team Scheduler")

# Inicializácia premenných pre neskoršie použitie
start_date = date.today()
end_date = date.today() + timedelta(days=30)
employees = []
teams = []
shift_types = []
collaborations = []
schedule_df = pd.DataFrame()

# Sidebar pre globálne nastavenia
with st.sidebar:
    st.header("⚙️ Globálne nastavenia")
    consider_skills = st.checkbox("Zohľadniť zručnosti", value=True)
    balance_workload = st.checkbox("Vyvážiť pracovnú záťaž", value=True)
    minimize_cost = st.checkbox("Minimalizovať náklady", value=False)

    st.header("🎯 Optimalizačné ciele")
    optimization_goal = st.selectbox(
        "Hlavný cieľ optimalizácie",
        ["Vyváženie záťaže", "Minimalizácia nákladov", "Maximalizácia spokojnosti", "Splnenie cieľových hodín"]
    )

    # PRIDANÉ: Databázové kontroly
    add_database_controls(st)

    st.header("📊 Export/Import")

    # Modifikované tlačidlá pre import/export s databázou
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Import z DB"):
            with st.spinner("Načítavam z databázy..."):
                if load_from_database(st):
                    st.success("✅ Dáta načítané")
                    st.rerun()

    with col2:
        if st.button("📤 Export do DB"):
            with st.spinner("Ukladám do databázy..."):
                if sync_to_database(st):
                    st.success("✅ Dáta uložené")

# Hlavné tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "⏰ Obdobie & Smeny",
    "🏢 Tímy & Požiadavky",
    "👥 Zamestnanci",
    "🏖️ Dovolenky",
    "🤝 Spolupráca",
    "📊 Generovanie",
    "📈 Analýzy",
    "💾 Databáza"
])
with tab1:
    st.subheader("📅 Plánovacie obdobie")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Začiatok", date.today())
    with col2:
        end_date = st.date_input("Koniec", date.today() + timedelta(days=30))

    if end_date < start_date:
        st.error("Koniec nesmie byť pred začiatkom!")
        st.stop()

    # Počet dní a základné info
    total_days = (end_date - start_date).days + 1
    weekdays = sum(1 for d in range(total_days) if (start_date + timedelta(d)).weekday() < 5)
    weekends = total_days - weekdays

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Celkom dní", total_days)
    with col2:
        st.metric("Pracovné dni", weekdays)
    with col3:
        st.metric("Víkendové dni", weekends)

    st.subheader("⏰ Definícia smien")

    # Inicializácia session state pre smeny
    if 'shifts' not in st.session_state:
        st.session_state.shifts = [
            {
                "name": "Denná", "start": time(6), "end": time(14), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.0,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            },
            {
                "name": "Poobedná", "start": time(14), "end": time(22), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.1,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            },
            {
                "name": "Nočná", "start": time(22), "end": time(6), "rest_days": 1,
                "skills": ["Bezpečnosť"], "min_level": "Pokročilý", "premium": 0.25,
                "weekend_applicable": True, "max_consecutive": 3, "min_employees": 1, "max_employees": 2
            }
        ]

    # Náhrada pre riadky 660-690 v rn.py (tab1 - smeny sekcia)

    # Zabezpečenie spätnej kompatibility - pridanie chýbajúcich kľúčov
    for i, shift_data in enumerate(st.session_state.shifts):
        # Pridanie chýbajúcich kľúčov s defaultnými hodnotami
        default_values = {
            "weekend_applicable": True,
            "max_consecutive": 5,
            "min_employees": 1,
            "max_employees": 3,
            "premium": 0.0,
            "rest_days": 0,
            "skills": [],
            "min_level": "Zaciatocník"
        }

        for key, default_value in default_values.items():
            if key not in shift_data:
                st.session_state.shifts[i][key] = default_value

    # Správa smien
    shift_types = []
    for i, shift_data in enumerate(st.session_state.shifts):
        with st.expander(f"Smena: {shift_data['name']}", expanded=i == 0):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                name = st.text_input("Názov", shift_data['name'], key=f"shift_name_{i}")

                # Bezpečná konverzia time hodnôt
                start_val = shift_data.get('start')
                if not isinstance(start_val, time):
                    if start_val is None:
                        start_val = time(6, 0)
                    elif isinstance(start_val, str):
                        try:
                            start_val = datetime.strptime(start_val, '%H:%M:%S').time()
                        except:
                            start_val = time(6, 0)
                    elif isinstance(start_val, timedelta):
                        total_seconds = int(start_val.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        start_val = time(hours % 24, minutes)
                    else:
                        start_val = time(6, 0)

                end_val = shift_data.get('end')
                if not isinstance(end_val, time):
                    if end_val is None:
                        end_val = time(14, 0)
                    elif isinstance(end_val, str):
                        try:
                            end_val = datetime.strptime(end_val, '%H:%M:%S').time()
                        except:
                            end_val = time(14, 0)
                    elif isinstance(end_val, timedelta):
                        total_seconds = int(end_val.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        end_val = time(hours % 24, minutes)
                    else:
                        end_val = time(14, 0)

                start_time = st.time_input("Začiatok", start_val, key=f"shift_start_{i}")
                end_time = st.time_input("Koniec", end_val, key=f"shift_end_{i}")

            with col2:
                rest_days = st.number_input("Dni voľna po smene", 0, 7, int(shift_data['rest_days']),
                                            key=f"shift_rest_{i}")

                # Oprava pre mixed numeric types
                premium_val = float(shift_data.get('premium', 0.0))
                premium = st.number_input("Prémia (%)", 0.0, 1.0, premium_val, step=0.05,
                                          key=f"shift_premium_{i}")
                weekend_applicable = st.checkbox("Platí aj na víkendy", shift_data['weekend_applicable'],
                                                 key=f"shift_weekend_{i}")

            with col3:
                min_employees = st.number_input("Min. zamestnancov", 1, 10, int(shift_data['min_employees']),
                                                key=f"shift_min_{i}")
                max_employees = st.number_input("Max. zamestnancov", 1, 20, int(shift_data['max_employees']),
                                                key=f"shift_max_{i}")
                max_consecutive = st.number_input("Max. po sebe idúcich dní", 1, 14, int(shift_data['max_consecutive']),
                                                  key=f"shift_consec_{i}")

            with col4:
                skills = st.multiselect("Požadované zručnosti",
                                        ["Prvá pomoc", "Vedenie tímu", "Technické zručnosti", "Komunikácia",
                                         "Bezpečnosť"],
                                        shift_data['skills'], key=f"shift_skills_{i}")
                min_level = st.selectbox("Min. úroveň",
                                         ["Zaciatocník", "Pokročilý", "Expert", "Supervízor"],
                                         index=["Zaciatocník", "Pokročilý", "Expert", "Supervízor"].index(
                                             shift_data['min_level']),
                                         key=f"shift_level_{i}")

            # Aktualizácia session state
            st.session_state.shifts[i] = {
                "name": name, "start": start_time, "end": end_time, "rest_days": rest_days,
                "skills": skills, "min_level": min_level, "premium": premium,
                "weekend_applicable": weekend_applicable, "max_consecutive": max_consecutive,
                "min_employees": min_employees, "max_employees": max_employees
            }

            shift_types.append(ShiftType(
                name=name, start=start_time, end=end_time, rest_days_after=rest_days,
                required_skills=skills, min_skill_level=SkillLevel(min_level), premium_pay=premium,
                is_weekend_applicable=weekend_applicable, max_consecutive_days=max_consecutive,
                min_employees=min_employees, max_employees=max_employees
            ))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať smenu"):
            st.session_state.shifts.append({
                "name": "Nová smena", "start": time(9), "end": time(17), "rest_days": 0,
                "skills": [], "min_level": "Zaciatocník", "premium": 0.0,
                "weekend_applicable": True, "max_consecutive": 5, "min_employees": 1, "max_employees": 3
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť poslednú smenu") and len(st.session_state.shifts) > 1:
            st.session_state.shifts.pop()
            st.rerun()

with tab2:
    st.subheader("🏢 Správa tímov a firemných požiadaviek")

    # Inicializácia session state pre tímy
    if 'teams' not in st.session_state:
        st.session_state.teams = [
            {
                "id": "TEAM001", "name": "Prevádzkový tím", "description": "Hlavný prevádzkový tím",
                "priority": 1, "color": "#4CAF50", "department": "Výroba", "location": "Bratislava",
                "min_coverage": {"Denná": 2, "Poobedná": 2, "Nočná": 1},
                "max_coverage": {"Denná": 4, "Poobedná": 4, "Nočná": 2},
                "target_coverage": {"Denná": 3, "Poobedná": 3, "Nočná": 1},
                "weekend_multiplier": 1.0, "holiday_multiplier": 0.5,
                "supervisor_required": False, "emergency_contact": True
            }
        ]

    # Zabezpečenie spätnej kompatibility pre tímy
    for i, team_data in enumerate(st.session_state.teams):
        default_team_values = {
            "department": "",
            "location": "",
            "min_coverage": {},
            "max_coverage": {},
            "target_coverage": {},
            "weekend_multiplier": 1.0,
            "holiday_multiplier": 0.5,
            "supervisor_required": False,
            "emergency_contact": True
        }

        for key, default_value in default_team_values.items():
            if key not in team_data:
                st.session_state.teams[i][key] = default_value

    teams = []
    for i, team_data in enumerate(st.session_state.teams):
        with st.expander(f"Tím: {team_data['name']}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                team_id = st.text_input("ID tímu", team_data['id'], key=f"team_id_{i}")
                name = st.text_input("Názov tímu", team_data['name'], key=f"team_name_{i}")
                description = st.text_area("Popis", team_data['description'], key=f"team_desc_{i}")
                department = st.text_input("Oddelenie", team_data.get('department', ''), key=f"team_dept_{i}")
                location = st.text_input("Lokalita", team_data.get('location', ''), key=f"team_loc_{i}")

            with col2:
                priority = st.number_input("Priorita", 1, 10, team_data['priority'], key=f"team_priority_{i}")
                color = st.color_picker("Farba", team_data['color'], key=f"team_color_{i}")

            # Firemné požiadavky na pokrytie
            st.write("**Požiadavky na pokrytie smien:**")
            col1, col2, col3 = st.columns(3)

            min_coverage = {}
            max_coverage = {}
            target_coverage = {}

            with col1:
                st.write("**Minimum:**")
                for shift in shift_types:
                    min_val = st.number_input(
                        f"Min {shift.name}", 0, 10,
                        team_data.get('min_coverage', {}).get(shift.name, 1),
                        key=f"team_min_{i}_{shift.name}"
                    )
                    min_coverage[shift.name] = min_val

            with col2:
                st.write("**Cieľ:**")
                for shift in shift_types:
                    target_val = st.number_input(
                        f"Cieľ {shift.name}", 0, 15,
                        team_data.get('target_coverage', {}).get(shift.name, 1),
                        key=f"team_target_{i}_{shift.name}"
                    )
                    target_coverage[shift.name] = target_val

            with col3:
                st.write("**Maximum:**")
                for shift in shift_types:
                    max_val = st.number_input(
                        f"Max {shift.name}", 0, 20,
                        team_data.get('max_coverage', {}).get(shift.name, 2),
                        key=f"team_max_{i}_{shift.name}"
                    )
                    max_coverage[shift.name] = max_val

            # Pokročilé nastavenia
            with st.expander("Pokročilé nastavenia tímu"):
                weekend_multiplier = st.number_input("Víkendový násobiteľ pokrytia", 0.1, 2.0, 1.0, step=0.1,
                                                     key=f"team_weekend_{i}")
                holiday_multiplier = st.number_input("Sviatkový násobiteľ pokrytia", 0.1, 2.0, 0.5, step=0.1,
                                                     key=f"team_holiday_{i}")
                supervisor_required = st.checkbox("Vždy vyžadovať supervízora", key=f"team_supervisor_{i}")
                emergency_contact = st.checkbox("Vyžadovať pohotovostný kontakt", True, key=f"team_emergency_{i}")

            # Aktualizácia session state
            st.session_state.teams[i] = {
                "id": team_id, "name": name, "description": description,
                "priority": priority, "color": color, "department": department, "location": location,
                "min_coverage": min_coverage, "max_coverage": max_coverage, "target_coverage": target_coverage,
                "weekend_multiplier": weekend_multiplier, "holiday_multiplier": holiday_multiplier,
                "supervisor_required": supervisor_required, "emergency_contact": emergency_contact
            }

            # Vytvorenie Company Requirements
            company_req = CompanyRequirements(
                min_coverage_per_shift=min_coverage,
                max_coverage_per_shift=max_coverage,
                target_coverage_per_shift=target_coverage,
                weekend_multiplier=weekend_multiplier,
                holiday_multiplier=holiday_multiplier,
                supervisor_always_present=supervisor_required,
                emergency_contact_required=emergency_contact
            )

            teams.append(Team(
                id=team_id, name=name, description=description, priority=priority,
                color=color, department=department, location=location,
                company_requirements=company_req
            ))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať tím"):
            new_id = f"TEAM{len(st.session_state.teams) + 1:03d}"
            st.session_state.teams.append({
                "id": new_id, "name": "Nový tím", "description": "", "priority": 1, "color": "#FF9800",
                "department": "", "location": "", "min_coverage": {}, "max_coverage": {}, "target_coverage": {}
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť posledný tím") and len(st.session_state.teams) > 1:
            st.session_state.teams.pop()
            st.rerun()

with tab3:
    st.subheader("👥 Správa zamestnancov")

    # Inicializácia session state pre zamestnancov
    if 'employees' not in st.session_state:
        st.session_state.employees = [
            {
                "id": "EMP001", "name": "Ján Novák", "team_id": "TEAM001", "max_cons": 5, "max_night": 8,
                "hourly_rate": 15.0, "skills": [], "monthly_target": 160, "weekly_min": 20, "weekly_max": 48,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 4.0, "seniority": 2, "email": "", "phone": "", "preferences": []
            },
            {
                "id": "EMP002", "name": "Mária Svobodová", "team_id": "TEAM001", "max_cons": 4, "max_night": 6,
                "hourly_rate": 18.0, "skills": [], "monthly_target": 160, "weekly_min": 30, "weekly_max": 45,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 4.5, "seniority": 5, "email": "", "phone": "", "preferences": []
            },
            {
                "id": "EMP003", "name": "Peter Kováč", "team_id": "TEAM001", "max_cons": 6, "max_night": 10,
                "hourly_rate": 20.0, "skills": [], "monthly_target": 170, "weekly_min": 25, "weekly_max": 50,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 3.5, "seniority": 1, "email": "", "phone": "", "preferences": []
            }
        ]

    # Náhrada pre tab3 (zamestnanci) v rn.py - oprava numeric types

    # Zabezpečenie spätnej kompatibility pre zamestnancov
    for i, emp_data in enumerate(st.session_state.employees):
        default_emp_values = {
            "monthly_target": 160,
            "weekly_min": 20,
            "weekly_max": 48,
            "overtime_eligible": True,
            "weekend_allowed": True,
            "night_restriction": False,
            "performance": 3.0,
            "seniority": 0,
            "email": "",
            "phone": "",
            "preferences": [],
            "skills": []
        }

        for key, default_value in default_emp_values.items():
            if key not in emp_data:
                st.session_state.employees[i][key] = default_value

    employees = []
    team_options = {team["id"]: team["name"] for team in st.session_state.teams}

    for i, emp_data in enumerate(st.session_state.employees):
        with st.expander(f"Zamestnanec: {emp_data['name']}", expanded=False):

            # Základné informácie
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Základné údaje:**")
                emp_id = st.text_input("ID", emp_data['id'], key=f"emp_id_{i}")
                name = st.text_input("Meno a priezvisko", emp_data['name'], key=f"emp_name_{i}")
                team_id = st.selectbox("Tím", list(team_options.keys()),
                                       index=list(team_options.keys()).index(emp_data['team_id']) if emp_data[
                                                                                                         'team_id'] in team_options else 0,
                                       format_func=lambda x: team_options[x], key=f"emp_team_{i}")
                email = st.text_input("Email", emp_data.get('email', ''), key=f"emp_email_{i}")
                phone = st.text_input("Telefón", emp_data.get('phone', ''), key=f"emp_phone_{i}")

            with col2:
                st.write("**Pracovné podmienky:**")
                monthly_target = st.number_input("Mesačný cieľ hodín", 80, 200,
                                                 int(emp_data.get('monthly_target', 160)),
                                                 key=f"emp_monthly_{i}")
                weekly_min = st.number_input("Min. týždenných hodín", 10, 40,
                                             int(emp_data.get('weekly_min', 20)),
                                             key=f"emp_weekly_min_{i}")
                weekly_max = st.number_input("Max. týždenných hodín", 30, 60,
                                             int(emp_data.get('weekly_max', 48)),
                                             key=f"emp_weekly_max_{i}")
                max_cons = st.number_input("Max. po sebe idúcich dní", 1, 14,
                                           int(emp_data.get('max_cons', 5)),
                                           key=f"emp_cons_{i}")
                max_night = st.number_input("Max. nočných smien", 0, 20,
                                            int(emp_data.get('max_night', 8)),
                                            key=f"emp_night_{i}")

            with col3:
                st.write("**Finančné a osobné:**")
                hourly_rate = st.number_input("Hodinová sadzba (€)", 10.0, 100.0,
                                              float(emp_data.get('hourly_rate', 15.0)),
                                              step=0.5, key=f"emp_rate_{i}")
                performance = st.number_input("Hodnotenie výkonu (1-5)", 1.0, 5.0,
                                              float(emp_data.get('performance', 3.0)),
                                              step=0.5, key=f"emp_perf_{i}")
                seniority = st.number_input("Roky stáže", 0, 40,
                                            int(emp_data.get('seniority', 0)),
                                            key=f"emp_senior_{i}")
                annual_vacation = st.number_input("Ročný nárok dovolenky", 20, 35, 25,
                                                  key=f"emp_vacation_{i}")

            # Obmedzenia a možnosti
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pracovné možnosti:**")
                contract_type = st.selectbox("Typ zmluvy",
                                             ["Plný úväzok", "Čiastočný úväzok", "Dočasný", "Zmluvný"],
                                             key=f"emp_contract_{i}")
                overtime_eligible = st.checkbox("Môže robiť nadčasy",
                                                bool(emp_data.get('overtime_eligible', True)),
                                                key=f"emp_overtime_{i}")
                weekend_allowed = st.checkbox("Môže pracovať cez víkend",
                                              bool(emp_data.get('weekend_allowed', True)),
                                              key=f"emp_weekend_{i}")
                night_restriction = st.checkbox("Zákaz nočných smien",
                                                bool(emp_data.get('night_restriction', False)),
                                                key=f"emp_night_restrict_{i}")

            with col2:
                st.write("**Zručnosti:**")
                available_skills = ["Prvá pomoc", "Vedenie tímu", "Technické zručnosti", "Komunikácia", "Bezpečnosť",
                                    "Jazykové", "IT"]
                employee_skills = []
                for skill_name in available_skills:
                    if st.checkbox(f"{skill_name}", key=f"emp_skill_{i}_{skill_name}"):
                        level = st.selectbox(f"Úroveň {skill_name}",
                                             ["Zaciatocník", "Pokročilý", "Expert", "Supervízor"],
                                             key=f"emp_skill_level_{i}_{skill_name}")
                        employee_skills.append(Skill(name=skill_name, level=SkillLevel(level)))

            # Preferencie smien
            st.write("**Preferencie smien:**")
            shift_names = [s["name"] for s in st.session_state.shifts]
            preferences = st.multiselect("Preferované smeny (prázdne = všetky)",
                                         shift_names,
                                         emp_data.get('preferences', []),
                                         key=f"emp_prefs_{i}")

            # Aktualizácia session state
            st.session_state.employees[i] = {
                "id": emp_id, "name": name, "team_id": team_id, "max_cons": max_cons,
                "max_night": max_night, "hourly_rate": hourly_rate, "skills": employee_skills,
                "monthly_target": monthly_target, "weekly_min": weekly_min, "weekly_max": weekly_max,
                "overtime_eligible": overtime_eligible, "weekend_allowed": weekend_allowed,
                "night_restriction": night_restriction, "performance": performance, "seniority": seniority,
                "email": email, "phone": phone, "preferences": preferences
            }

            # Vytvorenie Work Requirements
            work_req = WorkRequirement(
                monthly_hours_target=monthly_target,
                weekly_hours_min=weekly_min,
                weekly_hours_max=weekly_max,
                max_consecutive_work_days=max_cons
            )

            employees.append(Employee(
                id=emp_id, name=name, team_id=team_id, max_consecutive_days=max_cons,
                max_night_shifts=max_night, hourly_rate=hourly_rate, skills=employee_skills,
                contract_type=ContractType(contract_type), work_requirements=work_req,
                overtime_eligible=overtime_eligible, weekend_work_allowed=weekend_allowed,
                night_shift_restriction=night_restriction, seniority_years=seniority,
                performance_rating=performance, email=email, phone=phone,
                preferences=preferences, annual_vacation_days=annual_vacation
            ))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Pridať zamestnanca"):
            new_id = f"EMP{len(st.session_state.employees) + 1:03d}"
            st.session_state.employees.append({
                "id": new_id, "name": "Nový zamestnanec",
                "team_id": list(team_options.keys())[0] if team_options else "TEAM001",
                "max_cons": 5, "max_night": 8, "hourly_rate": 15.0, "skills": [],
                "monthly_target": 160, "weekly_min": 20, "weekly_max": 48,
                "overtime_eligible": True, "weekend_allowed": True, "night_restriction": False,
                "performance": 3.0, "seniority": 0, "email": "", "phone": "", "preferences": []
            })
            st.rerun()

    with col2:
        if st.button("🗑️ Odstrániť posledného"):
            if len(st.session_state.employees) > 1:
                st.session_state.employees.pop()
                st.rerun()

with tab4:
    st.subheader("🏖️ Správa dovoleniek a neprítomností")

    # Inicializácia session state pre dovolenky
    if 'vacation_requests' not in st.session_state:
        st.session_state.vacation_requests = {}

    # Výber zamestnanca pre správu dovolenky
    employee_names = {emp["id"]: emp["name"] for emp in st.session_state.employees}
    selected_emp_id = st.selectbox("Vyberte zamestnanca:", list(employee_names.keys()),
                                   format_func=lambda x: employee_names[x])

    if selected_emp_id:
        selected_emp = next(emp for emp in st.session_state.employees if emp["id"] == selected_emp_id)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Dovolenka pre: {selected_emp['name']}**")

            # Inicializácia dovoleniek pre zamestnanca
            if selected_emp_id not in st.session_state.vacation_requests:
                st.session_state.vacation_requests[selected_emp_id] = []

            # Nová žiadosť o dovolenku
            with st.expander("➕ Nová žiadosť o dovolenku", expanded=True):
                vacation_start = st.date_input("Začiatok", key=f"vac_start_{selected_emp_id}")
                vacation_end = st.date_input("Koniec", key=f"vac_end_{selected_emp_id}")
                vacation_type = st.selectbox("Typ neprítomnosti",
                                             ["Riadna dovolenka", "Nemocenská", "Osobné voľno", "Materská/otcovská",
                                              "Neplatené voľno", "Školenie", "Náhradné voľno"],
                                             key=f"vac_type_{selected_emp_id}")
                vacation_reason = st.text_area("Dôvod/Poznámka", key=f"vac_reason_{selected_emp_id}")
                vacation_priority = st.selectbox("Priorita", ["Nízka", "Stredná", "Vysoká", "Kritická"],
                                                 index=1, key=f"vac_priority_{selected_emp_id}")

                if st.button("Pridať žiadosť", key=f"add_vac_{selected_emp_id}"):
                    if vacation_end >= vacation_start:
                        duration = (vacation_end - vacation_start).days + 1
                        new_request = {
                            "start_date": vacation_start,
                            "end_date": vacation_end,
                            "vacation_type": vacation_type,
                            "reason": vacation_reason,
                            "priority": vacation_priority,
                            "approved": False,
                            "duration": duration,
                            "created_date": date.today()
                        }
                        st.session_state.vacation_requests[selected_emp_id].append(new_request)
                        st.success(f"Žiadosť pridaná! ({duration} dní)")
                        st.rerun()
                    else:
                        st.error("Koniec nemôže byť pred začiatkom!")

        with col2:
            st.write("**Prehľad dovolenky:**")

            # Štatistiky dovolenky
            annual_entitlement = selected_emp.get('annual_vacation', 25)
            approved_annual = sum(
                req["duration"] for req in st.session_state.vacation_requests.get(selected_emp_id, [])
                if req["approved"] and req["vacation_type"] == "Riadna dovolenka"
            )
            pending_annual = sum(
                req["duration"] for req in st.session_state.vacation_requests.get(selected_emp_id, [])
                if not req["approved"] and req["vacation_type"] == "Riadna dovolenka"
            )
            remaining = annual_entitlement - approved_annual

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Ročný nárok", annual_entitlement)
            with col_b:
                st.metric("Využité", approved_annual)
            with col_c:
                st.metric("Zostatok", remaining)

            if pending_annual > 0:
                st.info(f"Čakajúce žiadosti: {pending_annual} dní")

            # Upozornenia
            if remaining < 0:
                st.error("⚠️ Prekročený ročný nárok!")
            elif remaining < 5:
                st.warning("⚠️ Nízky zostatok dovolenky!")

        # Zoznam žiadostí
        st.subheader("📋 Existujúce žiadosti")

        vacation_requests = st.session_state.vacation_requests.get(selected_emp_id, [])
        if vacation_requests:
            for idx, req in enumerate(vacation_requests):
                with st.expander(
                        f"{req['vacation_type']} | {req['start_date']} - {req['end_date']} ({req['duration']} dní)"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Typ:** {req['vacation_type']}")
                        st.write(f"**Obdobie:** {req['start_date']} - {req['end_date']}")
                        st.write(f"**Dĺžka:** {req['duration']} dní")
                        st.write(f"**Priorita:** {req['priority']}")

                    with col2:
                        st.write(f"**Dôvod:** {req['reason']}")
                        st.write(f"**Vytvorené:** {req['created_date']}")
                        status = "✅ Schválené" if req['approved'] else "⏳ Čaká na schválenie"
                        st.write(f"**Status:** {status}")

                    with col3:
                        if not req['approved']:
                            if st.button("✅ Schváliť", key=f"approve_{selected_emp_id}_{idx}"):
                                st.session_state.vacation_requests[selected_emp_id][idx]['approved'] = True
                                st.success("Žiadosť schválená!")
                                st.rerun()
                        else:
                            if st.button("❌ Zrušiť schválenie", key=f"unapprove_{selected_emp_id}_{idx}"):
                                st.session_state.vacation_requests[selected_emp_id][idx]['approved'] = False
                                st.success("Schválenie zrušené!")
                                st.rerun()

                        if st.button("🗑️ Odstrániť", key=f"delete_vac_{selected_emp_id}_{idx}"):
                            st.session_state.vacation_requests[selected_emp_id].pop(idx)
                            st.success("Žiadosť odstránená!")
                            st.rerun()
        else:
            st.info("Žiadne žiadosti o dovolenku")

    # Kalendárny prehľad dovoleniek
    st.subheader("📅 Kalendárny prehľad dovoleniek")

    # Vytvoríme prehľad všetkých dovoleniek
    vacation_overview = []
    for emp_id, requests in st.session_state.vacation_requests.items():
        emp_name = employee_names.get(emp_id, f"EMP {emp_id}")
        for req in requests:
            if req['approved']:
                vacation_overview.append({
                    "Zamestnanec": emp_name,
                    "Typ": req['vacation_type'],
                    "Začiatok": req['start_date'],
                    "Koniec": req['end_date'],
                    "Dni": req['duration'],
                    "Dôvod": req['reason']
                })

    if vacation_overview:
        vacation_df = pd.DataFrame(vacation_overview)
        vacation_df = vacation_df.sort_values('Začiatok')
        st.dataframe(vacation_df, use_container_width=True)

        # Export dovoleniek
        csv_vacation = vacation_df.to_csv(index=False)
        st.download_button(
            "📥 Export dovoleniek (CSV)",
            csv_vacation,
            "dovolenky.csv",
            "text/csv"
        )
    else:
        st.info("Žiadne schválené dovolenky")

with tab5:
    st.subheader("🤝 Spolupráca medzi tímami")

    # Inicializácia session state pre spoluprácu
    if 'collaborations' not in st.session_state:
        st.session_state.collaborations = []

    collaborations = []  # Inicializácia pre tento tab

    if len(teams) < 2:
        st.info("Pre nastavenie spolupráce potrebujete aspoň 2 tímy.")
    else:
        for i, collab_data in enumerate(st.session_state.collaborations):
            with st.expander(f"Spolupráca {i + 1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    team1_options = {t.id: t.name for t in teams}
                    team2_options = {t.id: t.name for t in teams}

                    team1_id = st.selectbox("Prvý tím", list(team1_options.keys()),
                                            format_func=lambda x: team1_options[x], key=f"collab_team1_{i}")
                    team2_id = st.selectbox("Druhý tím", list(team2_options.keys()),
                                            format_func=lambda x: team2_options[x], key=f"collab_team2_{i}")

                with col2:
                    max_shared = st.number_input("Max. zdieľaných zamestnancov", 1, 5, 2, key=f"collab_shared_{i}")
                    priority = st.number_input("Priorita spolupráce", 1, 10, 1, key=f"collab_priority_{i}")

                shift_names = [s.name for s in shift_types]
                allowed_shifts = st.multiselect("Povolené smeny pre spoluprácu", shift_names, shift_names,
                                                key=f"collab_shifts_{i}")

                collaborations.append(Collaboration(
                    team1_id=team1_id, team2_id=team2_id, shift_types=allowed_shifts,
                    max_shared_employees=max_shared, priority=priority
                ))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Pridať spoluprácu"):
                st.session_state.collaborations.append({})
                st.rerun()

        with col2:
            if st.button("🗑️ Odstrániť spoluprácu") and st.session_state.collaborations:
                st.session_state.collaborations.pop()
                st.rerun()

with tab6:
    st.subheader("📊 Generovanie plánu")

    # Nastavenie pokrytia automaticky z tímov
    st.subheader("🎯 Automatické pokrytie z tímových požiadaviek")
    coverage = {}

    for team in teams:
        coverage[team.id] = team.company_requirements.target_coverage_per_shift

    # Zobrazenie súhrnu pokrytia
    coverage_summary = []
    for team in teams:
        for shift_name, count in coverage.get(team.id, {}).items():
            coverage_summary.append({
                "Tím": team.name,
                "Smena": shift_name,
                "Cieľové pokrytie": count,
                "Min. pokrytie": team.company_requirements.min_coverage_per_shift.get(shift_name, count),
                "Max. pokrytie": team.company_requirements.max_coverage_per_shift.get(shift_name, count + 1)
            })

    if coverage_summary:
        coverage_df = pd.DataFrame(coverage_summary)
        st.dataframe(coverage_df, use_container_width=True)

    # Možnosť manuálneho prepísania
    st.subheader("⚙️ Manuálne úpravy pokrytia (voliteľné)")
    manual_override = st.checkbox("Povoliť manuálne úpravy pokrytia")

    if manual_override:
        for team in teams:
            st.write(f"**{team.name}** ({team.id})")
            coverage[team.id] = {}

            cols = st.columns(len(shift_types))
            for i, shift in enumerate(shift_types):
                with cols[i]:
                    original_count = team.company_requirements.target_coverage_per_shift.get(shift.name, 1)
                    count = st.number_input(
                        f"{shift.name}",
                        min_value=0, max_value=10, value=original_count,
                        key=f"manual_coverage_{team.id}_{shift.name}"
                    )
                    coverage[team.id][shift.name] = count

    # Aktualizácia dovoleniek do employee objektov
    for emp in employees:
        emp.vacation_requests = []
        if emp.id in st.session_state.vacation_requests:
            for req_data in st.session_state.vacation_requests[emp.id]:
                if req_data['approved']:
                    vacation_req = VacationRequest(
                        start_date=req_data['start_date'],
                        end_date=req_data['end_date'],
                        vacation_type=VacationType(req_data['vacation_type']),
                        reason=req_data['reason'],
                        approved=True,
                        priority=Priority(req_data['priority'])
                    )
                    emp.vacation_requests.append(vacation_req)

    # Diagnostika
    st.subheader("🔍 Diagnostika plánu")

    total_shifts_needed = 0
    total_employees = len(employees)
    period_days = (end_date - start_date).days + 1

    for team_id, team_coverage in coverage.items():
        for shift_name, count in team_coverage.items():
            total_shifts_needed += count * period_days

    avg_shifts_per_employee = total_shifts_needed / total_employees if total_employees > 0 else 0
    total_hours_needed = sum(
        count * period_days * next(s.duration_hours() for s in shift_types if s.name == shift_name)
        for team_coverage in coverage.values()
        for shift_name, count in team_coverage.items()
    )
    avg_hours_per_employee = total_hours_needed / total_employees if total_employees > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Celkový počet smien", total_shifts_needed)
    with col2:
        st.metric("Počet zamestnancov", total_employees)
    with col3:
        st.metric("Priemerné smeny/zamestnanec", f"{avg_shifts_per_employee:.1f}")
    with col4:
        st.metric("Priemerné hodiny/zamestnanec", f"{avg_hours_per_employee:.1f}")

    # Upozornenia a diagnostika
    unavailable_days = 0
    for emp in employees:
        for req in emp.vacation_requests:
            if req.approved:
                unavailable_days += req.get_duration_days()

    if avg_hours_per_employee > 200:
        st.error("❌ Príliš vysoká záťaž! Zamestnanci budú preťažení.")
    elif avg_hours_per_employee > 180:
        st.warning("⚠️ Vysoká záťaž. Plán môže byť náročný na splnenie.")
    else:
        st.success("✅ Rozumná záťaž. Plán by mal byť dobre optimalizovateľný.")

    if unavailable_days > 0:
        st.info(f"ℹ️ Celkovo {unavailable_days} dní dovolenky/neprítomnosti")

    # Analýza dostupnosti zamestnancov
    st.subheader("👥 Analýza dostupnosti zamestnancov")
    availability_analysis = []

    for emp in employees:
        unavailable_count = sum(req.get_duration_days() for req in emp.vacation_requests if req.approved)
        available_days = period_days - unavailable_count
        availability_pct = (available_days / period_days) * 100

        target_hours = emp.work_requirements.monthly_hours_target
        max_possible_hours = available_days * 8  # Predpokladáme max 8h/deň

        availability_analysis.append({
            "Zamestnanec": emp.name,
            "Tím": next((t.name for t in teams if t.id == emp.team_id), "Neznámy"),
            "Dostupné dni": available_days,
            "Dostupnosť (%)": f"{availability_pct:.1f}%",
            "Cieľové hodiny": target_hours,
            "Max. možné hodiny": max_possible_hours,
            "Realizovateľnosť": "✅ OK" if max_possible_hours >= target_hours else "⚠️ Problém"
        })

    availability_df = pd.DataFrame(availability_analysis)
    st.dataframe(availability_df, use_container_width=True)

    # Generovanie plánu
    st.subheader("🚀 Generovanie plánu")

    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit = st.number_input("Časový limit (sekundy)", 60, 600, 180)
    with col2:
        fallback_enabled = st.checkbox("Povoliť automatické znižovanie požiadaviek pri zlyhaní", True)
    with col3:
        solver_mode = st.selectbox("Režim riešenia", ["Rýchly", "Vyvážený", "Presný"])

    if st.button("🧮 Vygenerovať plán", type="primary"):
        if not employees:
            st.error("❌ Musíte pridať aspoň jedného zamestnanca!")
            st.stop()

        if not teams:
            st.error("❌ Musíte pridať aspoň jeden tím!")
            st.stop()

        if not shift_types:
            st.error("❌ Musíte definovať aspoň jednu smenu!")
            st.stop()

        # Zabezpečíme že collaborations je definované
        if 'collaborations' not in locals():
            collaborations = []

        # Nastavenie optimalizačných cieľov podľa výberu
        if optimization_goal == "Minimalizácia nákladov":
            minimize_cost = True
            balance_workload = False
        elif optimization_goal == "Vyváženie záťaže":
            minimize_cost = False
            balance_workload = True
        else:
            minimize_cost = False
            balance_workload = True

        try:
            with st.spinner("Generujem pokročilý plán..."):
                scheduler = AdvancedScheduler(
                    employees=employees,
                    teams=teams,
                    shift_types=shift_types,
                    period_start=start_date,
                    period_end=end_date,
                    coverage=coverage,
                    collaborations=collaborations,
                    consider_skills=consider_skills,
                    balance_workload=balance_workload,
                    minimize_cost=minimize_cost
                )

                schedule_df = scheduler.solve(limit=time_limit)

        except Exception as exc:
            st.error(f"❌ Chyba pri generovaní plánu: {exc}")

            if fallback_enabled:
                st.info("🔄 Pokúšam sa s redukovanými požiadavkami...")
                try:
                    # Drastické zníženie pokrytia
                    reduced_coverage = {}
                    for team_id, team_coverage in coverage.items():
                        reduced_coverage[team_id] = {}
                        for shift_name, count in team_coverage.items():
                            # Znížime na minimum alebo 1
                            reduced_coverage[team_id][shift_name] = max(1, count // 2)

                    # Fallback scheduler s minimálnymi obmedzeniami
                    scheduler_fallback = AdvancedScheduler(
                        employees=employees,
                        teams=teams,
                        shift_types=shift_types,
                        period_start=start_date,
                        period_end=end_date,
                        coverage=reduced_coverage,
                        collaborations=[],  # Vypnúť collaborations
                        consider_skills=False,  # Vypnúť skill checking
                        balance_workload=False,  # Vypnúť balance
                        minimize_cost=False  # Len základná optimalizácia
                    )

                    schedule_df = scheduler_fallback.solve(limit=time_limit)
                    st.warning("⚠️ Plán bol vygenerovaný s minimálnymi požiadavkami!")

                except Exception as fallback_exc:
                    st.error(f"❌ Ani s minimálnymi požiadavkami sa nepodarilo vygenerovať plán: {fallback_exc}")

                    # Posledná šanca - úplne základný plán
                    st.info("🔄 Pokúšam sa s úplne základným plánom...")
                    try:
                        # Minimálne pokrytie - len 1 osoba na smenu
                        minimal_coverage = {}
                        for team_id in coverage.keys():
                            minimal_coverage[team_id] = {}
                            for shift_name in coverage[team_id].keys():
                                minimal_coverage[team_id][shift_name] = 1

                        scheduler_minimal = AdvancedScheduler(
                            employees=employees[:min(3, len(employees))],  # Max 3 zamestnanci
                            teams=teams[:1],  # Len prvý tím
                            shift_types=shift_types[:2],  # Len prvé 2 smeny
                            period_start=start_date,
                            period_end=min(end_date, start_date + timedelta(days=7)),  # Max týždeň
                            coverage=minimal_coverage,
                            collaborations=[],
                            consider_skills=False,
                            balance_workload=False,
                            minimize_cost=False
                        )

                        schedule_df = scheduler_minimal.solve(limit=60)
                        st.warning("⚠️ Vygenerovaný bol iba ukážkový základný plán!")

                    except Exception as final_exc:
                        st.error(f"❌ Nepodarilo sa vygenerovať ani základný plán: {final_exc}")
                        st.error("💡 Skúste:")
                        st.error("- Pridať viac zamestnancov")
                        st.error("- Znížiť počet požadovaných smien")
                        st.error("- Skrátiť plánovacie obdobie")
                        st.error("- Odstrániť preferencie zamestnancov")
                        st.stop()
            else:
                st.stop()

        st.success("✅ Pokročilý plán úspešne vygenerovaný!")

        # Zobrazenie výsledkov v rozšírených tabs
        tab_schedule, tab_summary, tab_teams, tab_costs, tab_vacations = st.tabs([
            "📋 Plán", "📊 Súhrn zamestnancov", "🏢 Súhrn tímov", "💰 Náklady", "🏖️ Dovolenky"
        ])

        with tab_schedule:
            st.subheader("📋 Rozvrh smien")

            # Kompletný kalendár pre všetkých zamestnancov
            if not schedule_df.empty:
                # Vytvoríme kompletný daterange pre celé obdobie
                full_date_range = pd.date_range(start=start_date, end=end_date)

                # Vytvoríme prázdnu tabuľku pre všetkých zamestnancov a všetky dni
                schedule_matrix = {}

                # Inicializujeme prázdnu maticu pre každého zamestnanca
                for emp in employees:
                    schedule_matrix[emp.name] = {
                        'Tím': next((t.name for t in teams if t.id == emp.team_id), "Neznámy")
                    }
                    # Pridáme každý deň ako stĺpec s defaultnou hodnotou "-"
                    for single_date in full_date_range:
                        day_key = single_date.strftime('%d.%m')
                        schedule_matrix[emp.name][day_key] = '-'

                # Naplníme skutočné smeny z schedule_df
                for _, row in schedule_df.iterrows():
                    emp_name = row['Zamestnanec']
                    date_obj = pd.to_datetime(row['Dátum']).strftime('%d.%m')
                    shift_name = row['Zmena']

                    # Mapovanie názvov smien na krátke kódy
                    shift_mapping = {
                        'Denná': 'D',
                        'Poobedná': 'P',
                        'Nočná': 'N'
                    }

                    # Použijeme mapovanie alebo prvé písmeno ako fallback
                    short_shift = shift_mapping.get(shift_name, shift_name[:1].upper() if shift_name else '-')

                    if emp_name in schedule_matrix:
                        schedule_matrix[emp_name][date_obj] = short_shift

                # Konvertujeme na DataFrame
                display_df = pd.DataFrame.from_dict(schedule_matrix, orient='index')

                # Pridáme súčty hodín a nákladov
                hours_summary = schedule_df.groupby('Zamestnanec')['Hodiny'].sum()
                cost_summary = schedule_df.groupby('Zamestnanec')['Náklady'].sum()

                display_df['Celkom hodín'] = display_df.index.map(hours_summary).fillna(0)
                display_df['Celkom nákladov (€)'] = display_df.index.map(cost_summary).fillna(0).round(2)

                # Pridáme informácie o cieľových hodinách
                target_hours_map = {emp.name: emp.work_requirements.monthly_hours_target for emp in employees}
                display_df['Cieľové hodiny'] = display_df.index.map(target_hours_map).fillna(160)
                display_df['Rozdiel od cieľa'] = display_df['Celkom hodín'] - display_df['Cieľové hodiny']

                # Zoradíme stĺpce - najprv Tím, potom dátumy, potom súčty
                date_columns = [col for col in display_df.columns if
                                '.' in col and 'Celkom' not in col and 'Cieľové' not in col and 'Rozdiel' not in col]
                date_columns_sorted = sorted(date_columns, key=lambda x: datetime.strptime(x + '.2025', '%d.%m.%Y'))

                column_order = ['Tím'] + date_columns_sorted + ['Celkom hodín', 'Cieľové hodiny', 'Rozdiel od cieľa',
                                                                'Celkom nákladov (€)']
                display_df = display_df[column_order]

                # Resetujeme index aby sa zamestnanec zobrazil ako stĺpec
                display_df.reset_index(inplace=True)
                display_df.rename(columns={'index': 'Zamestnanec'}, inplace=True)

                st.dataframe(display_df, use_container_width=True, height=500)

                # Legenda pre krátke názvy
                st.subheader("🔤 Legenda smien")

                # Vytvoríme mapovanie pre legendu
                shift_mapping = {
                    'Denná': 'D',
                    'Poobedná': 'P',
                    'Nočná': 'N'
                }

                legend_cols = st.columns(len(shift_types) + 1)  # +1 pre "-"
                for i, shift in enumerate(shift_types):
                    with legend_cols[i]:
                        short = shift_mapping.get(shift.name, shift.name[:1].upper())
                        hours = shift.duration_hours()
                        st.write(f"**{short}** = {shift.name} ({hours}h)")

                # Pridáme legendu pre voľný deň
                with legend_cols[-1]:
                    st.write(f"**-** = Voľný deň")

                # Export možnosti
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 Stiahnuť plán (CSV)",
                        csv_data,
                        "plan_smien_kompletny.csv",
                        "text/csv"
                    )

                with col2:
                    # Excel export s formátovaním
                    excel_csv = display_df.to_csv(index=False)
                    st.download_button(
                        "📊 Stiahnuť pre Excel",
                        excel_csv,
                        "plan_smien.csv",
                        "text/csv"
                    )

                with col3:
                    json_data = schedule_df.to_json(orient="records", date_format="iso")
                    st.download_button(
                        "📥 Stiahnuť detaily (JSON)",
                        json_data,
                        "plan_detaily.json",
                        "application/json"
                    )

                # Prehľad pokrytia po dňoch
                st.subheader("📊 Prehľad pokrytia")

                coverage_analysis = schedule_df.groupby(['Dátum', 'Zmena']).size().reset_index(name='Počet')
                coverage_pivot = coverage_analysis.pivot(index='Dátum', columns='Zmena', values='Počet').fillna(0)

                st.dataframe(coverage_pivot, use_container_width=True)

                # Kontrola nedostatočného pokrytia
                issues = []
                for team in teams:
                    for shift_name, required_count in coverage.get(team.id, {}).items():
                        if required_count > 0:
                            for single_date in full_date_range:
                                actual_count = len(schedule_df[
                                                       (pd.to_datetime(
                                                           schedule_df['Dátum']).dt.date == single_date.date()) &
                                                       (schedule_df['Zmena'] == shift_name) &
                                                       (schedule_df['Tím'] == team.name)
                                                       ])
                                if actual_count < required_count:
                                    issues.append({
                                        'Dátum': single_date.date(),
                                        'Tím': team.name,
                                        'Zmena': shift_name,
                                        'Požadované': required_count,
                                        'Skutočné': actual_count,
                                        'Chýba': required_count - actual_count
                                    })

                if issues:
                    st.warning("⚠️ Nájdené problémy s pokrytím:")
                    issues_df = pd.DataFrame(issues)
                    st.dataframe(issues_df, use_container_width=True)
                else:
                    st.success("✅ Všetky požiadavky na pokrytie sú splnené!")

            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_summary:
            st.subheader("📊 Rozšírený súhrn zamestnancov")
            summaries = scheduler.get_summary(schedule_df)

            if not summaries["employees"].empty:
                st.dataframe(summaries["employees"], use_container_width=True)

                # Grafy výkonnosti
                fig_data = summaries["employees"].copy()
                if "Odpracované hodiny" in fig_data.columns:
                    st.subheader("📈 Porovnanie odpracovaných vs cieľových hodín")
                    chart_data = fig_data[['Zamestnanec', 'Odpracované hodiny', 'Cieľové hodiny']].set_index(
                        'Zamestnanec')
                    st.bar_chart(chart_data)
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_teams:
            st.subheader("🏢 Rozšírený súhrn tímov")

            if not summaries["teams"].empty:
                st.dataframe(summaries["teams"], use_container_width=True)

                # Pie chart pre rozdelenie nákladov medzi tímami
                if "Celkom nákladov" in summaries["teams"].columns:
                    st.subheader("💰 Rozdelenie nákladov medzi tímami")
                    fig_data = summaries["teams"].set_index("Tím")["Celkom nákladov"]
                    st.bar_chart(fig_data)
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_costs:
            st.subheader("💰 Detailná analýza nákladov")

            if not schedule_df.empty:
                # Celkové náklady
                total_cost = schedule_df["Náklady"].sum()
                total_hours = schedule_df["Hodiny"].sum()
                avg_hourly = total_cost / total_hours if total_hours > 0 else 0
                weekend_cost = schedule_df[schedule_df["Je_víkend"] == True][
                    "Náklady"].sum() if "Je_víkend" in schedule_df.columns else 0
                regular_cost = total_cost - weekend_cost

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Celkové náklady", f"{total_cost:.2f} €")
                with col2:
                    st.metric("Víkendové náklady", f"{weekend_cost:.2f} €")
                with col3:
                    st.metric("Celkové hodiny", f"{total_hours:.1f}")
                with col4:
                    st.metric("Priemerná sadzba", f"{avg_hourly:.2f} €/h")

                # Náklady podľa tímov
                team_costs = schedule_df.groupby("Tím").agg({
                    "Náklady": "sum",
                    "Hodiny": "sum"
                }).reset_index()
                team_costs["Priemerná sadzba"] = team_costs["Náklady"] / team_costs["Hodiny"]

                st.subheader("Náklady podľa tímov")
                st.dataframe(team_costs, use_container_width=True)

                # Náklady podľa smien
                shift_costs = schedule_df.groupby("Zmena").agg({
                    "Náklady": ["sum", "mean"],
                    "Hodiny": "sum"
                }).reset_index()
                shift_costs.columns = ["Zmena", "Celkové náklady", "Priemerné náklady", "Celkové hodiny"]

                st.subheader("Náklady podľa smien")
                st.dataframe(shift_costs, use_container_width=True)

                # Graf nákladov
                st.subheader("📊 Vizualizácia nákladov")
                st.bar_chart(team_costs.set_index("Tím")["Náklady"])
            else:
                st.info("Žiadne dáta na zobrazenie.")

        with tab_vacations:
            st.subheader("🏖️ Analýza dovoleniek a súlad s plánom")

            if not summaries["vacations"].empty:
                st.dataframe(summaries["vacations"], use_container_width=True)

                # Analýza vplyvu dovoleniek na plán
                st.subheader("📊 Vplyv dovoleniek na pracovný plán")

                vacation_impact = []
                for emp in employees:
                    emp_schedule = schedule_df[
                        schedule_df["Zamestnanec"] == emp.name] if not schedule_df.empty else pd.DataFrame()
                    worked_days = len(emp_schedule)
                    total_days = period_days
                    vacation_days = sum(req.get_duration_days() for req in emp.vacation_requests if req.approved)
                    available_days = total_days - vacation_days

                    if available_days > 0:
                        utilization = (worked_days / available_days) * 100
                    else:
                        utilization = 0

                    vacation_impact.append({
                        "Zamestnanec": emp.name,
                        "Celkové dni": total_days,
                        "Dovolenkové dni": vacation_days,
                        "Dostupné dni": available_days,
                        "Odpracované dni": worked_days,
                        "Využitie (%)": f"{utilization:.1f}%",
                        "Status": "✅ Optimálne" if 70 <= utilization <= 90 else "⚠️ Kontrola potrebná"
                    })

                vacation_impact_df = pd.DataFrame(vacation_impact)
                st.dataframe(vacation_impact_df, use_container_width=True)
            else:
                st.info("Žiadne dáta o dovolenkách na zobrazenie.")

with tab7:
    st.subheader("📈 Pokročilé analýzy a reporty")

    if 'schedule_df' in locals() and not schedule_df.empty:

        # KPI Dashboard
        st.subheader("🎯 KPI Dashboard")

        # Výpočet KPI
        total_employees = len(employees)
        total_scheduled_hours = schedule_df["Hodiny"].sum()
        total_target_hours = sum(emp.work_requirements.monthly_hours_target for emp in employees)
        target_achievement = (total_scheduled_hours / total_target_hours * 100) if total_target_hours > 0 else 0

        unique_employees_scheduled = schedule_df["Zamestnanec"].nunique()
        employee_utilization = (unique_employees_scheduled / total_employees * 100) if total_employees > 0 else 0

        avg_satisfaction = 85.0  # Simulované - v reálnej aplikácii by sa počítalo z preferencií

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Splnenie cieľových hodín",
                f"{target_achievement:.1f}%",
                delta=f"{target_achievement - 100:.1f}%" if target_achievement != 0 else None
            )

        with col2:
            st.metric(
                "Využitie zamestnancov",
                f"{employee_utilization:.1f}%",
                delta=f"{employee_utilization - 100:.1f}%" if employee_utilization != 0 else None
            )

        with col3:
            total_cost = schedule_df["Náklady"].sum()
            budget_limit = 50000  # Simulovaný rozpočet
            budget_usage = (total_cost / budget_limit * 100) if budget_limit > 0 else 0
            st.metric(
                "Využitie rozpočtu",
                f"{budget_usage:.1f}%",
                delta=f"{budget_usage - 100:.1f}%" if budget_usage != 0 else None
            )

        with col4:
            st.metric("Spokojnosť zamestnancov", f"{avg_satisfaction:.1f}%")

        # Trendy a analýzy
        st.subheader("📊 Týždenné trendy")

        # Analýza po týždňoch
        schedule_with_week = schedule_df.copy()
        schedule_with_week['Dátum'] = pd.to_datetime(schedule_with_week['Dátum'])
        schedule_with_week['Týždeň'] = schedule_with_week['Dátum'].dt.isocalendar().week

        weekly_analysis = schedule_with_week.groupby('Týždeň').agg({
            'Hodiny': 'sum',
            'Náklady': 'sum',
            'Zamestnanec': 'nunique'
        }).reset_index()
        weekly_analysis.columns = ['Týždeň', 'Celkové hodiny', 'Celkové náklady', 'Aktívni zamestnanci']

        st.dataframe(weekly_analysis, use_container_width=True)

        # Graf týždenných trendov
        st.line_chart(weekly_analysis.set_index('Týždeň')[['Celkové hodiny', 'Aktívni zamestnanci']])

        # Analýza výkonnosti tímov
        st.subheader("🏢 Porovnanie výkonnosti tímov")

        team_performance = schedule_df.groupby('Tím').agg({
            'Hodiny': ['sum', 'mean'],
            'Náklady': ['sum', 'mean'],
            'Zamestnanec': 'nunique'
        }).reset_index()

        team_performance.columns = [
            'Tím', 'Celkové hodiny', 'Priemerné hodiny/smena',
            'Celkové náklady', 'Priemerné náklady/smena', 'Počet zamestnancov'
        ]

        # Výpočet efektivity (hodiny na zamestnanca)
        team_performance['Efektivita (h/zamestnanec)'] = team_performance['Celkové hodiny'] / team_performance[
            'Počet zamestnancov']

        st.dataframe(team_performance, use_container_width=True)

        # Identifikácia problémov a odporúčania
        st.subheader("⚠️ Identifikované problémy a odporúčania")

        problems = []
        recommendations = []

        # Kontrola preťažených zamestnancov
        employee_hours = schedule_df.groupby('Zamestnanec')['Hodiny'].sum()
        overworked = employee_hours[employee_hours > 200]
        if not overworked.empty:
            problems.append(f"Preťažení zamestnanci: {', '.join(overworked.index)}")
            recommendations.append("Prerozdeliť záťaž alebo pridať nových zamestnancov")

        # Kontrola nedostatočne využitých zamestnancov
        underutilized = employee_hours[employee_hours < 120]
        if not underutilized.empty:
            problems.append(f"Nedostatočne využití zamestnanci: {', '.join(underutilized.index)}")
            recommendations.append("Zvýšiť záťaž alebo prehodnotiť potrebu pozícií")

        # Kontrola vysokých nákladov
        if total_cost > budget_limit:
            problems.append(f"Prekročený rozpočet o {total_cost - budget_limit:.2f} €")
            recommendations.append("Optimalizovať rozloženie smien alebo znížiť prémiové hodiny")

        # Kontrola pokrytia víkendov
        weekend_coverage = schedule_df[schedule_df.get('Je_víkend', False) == True]
        if weekend_coverage.empty and any(shift.is_weekend_applicable for shift in shift_types):
            problems.append("Nedostatočné pokrytie víkendov")
            recommendations.append("Pridať víkendové smeny alebo motivovať zamestnancov prémií")

        if problems:
            st.warning("Identifikované problémy:")
            for i, problem in enumerate(problems, 1):
                st.write(f"{i}. {problem}")

            st.info("Odporúčania:")
            for i, recommendation in enumerate(recommendations, 1):
                st.write(f"{i}. {recommendation}")
        else:
            st.success("✅ Žiadne významné problémy neboli identifikované!")

        # Export pokročilých reportov
        st.subheader("📋 Export reportov")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Manažérsky report
            manager_report = {
                "KPI": {
                    "Splnenie_cielovych_hodin": f"{target_achievement:.1f}%",
                    "Vyuzitie_zamestnancov": f"{employee_utilization:.1f}%",
                    "Vyuzitie_rozpoctu": f"{budget_usage:.1f}%",
                    "Celkove_naklady": f"{total_cost:.2f} €"
                },
                "Problemy": problems,
                "Odporucania": recommendations
            }

            manager_json = json.dumps(manager_report, indent=2, ensure_ascii=False)
            st.download_button(
                "📊 Manažérsky report (JSON)",
                manager_json,
                "manager_report.json",
                "application/json"
            )

        with col2:
            # HR report
            hr_data = []
            for emp in employees:
                emp_schedule = schedule_df[schedule_df["Zamestnanec"] == emp.name]
                hr_data.append({
                    "Zamestnanec": emp.name,
                    "ID": emp.id,
                    "Tim": next((t.name for t in teams if t.id == emp.team_id), "Neznámy"),
                    "Odpracovane_hodiny": emp_schedule["Hodiny"].sum(),
                    "Cielove_hodiny": emp.work_requirements.monthly_hours_target,
                    "Naklady": emp_schedule["Náklady"].sum(),
                    "Pocet_smien": len(emp_schedule),
                    "Zostatok_dovolenky": emp.vacation_remaining()
                })

            hr_df = pd.DataFrame(hr_data)
            hr_csv = hr_df.to_csv(index=False)
            st.download_button(
                "👥 HR report (CSV)",
                hr_csv,
                "hr_report.csv",
                "text/csv"
            )

        with col3:
            # Finančný report
            financial_df = team_performance.copy()
            financial_csv = financial_df.to_csv(index=False)
            st.download_button(
                "💰 Finančný report (CSV)",
                financial_csv,
                "financial_report.csv",
                "text/csv"
            )

    else:
        st.info("Pre zobrazenie analýz najprv vygenerujte plán v záložke 'Generovanie'")

# Footer
st.markdown("---")
st.markdown("**PlanME Pro** - Enterprise Team Scheduler | Verzia 3.0")
st.markdown("💡 *Pokročilý plánovač s podporou dovoleniek, firemných požiadaviek a detailných analýz*")

# Sidebar s dodatočnými informáciami
with st.sidebar:
    st.markdown("---")
    st.subheader("📋 Rýchly prehľad")

    if 'employees' in locals():
        st.write(f"👥 Zamestnanci: {len(employees)}")
    if 'teams' in locals():
        st.write(f"🏢 Tímy: {len(teams)}")
    if 'shift_types' in locals():
        st.write(f"⏰ Smeny: {len(shift_types)}")

    # Počet dní plánovania
    if 'start_date' in locals() and 'end_date' in locals():
        total_period_days = (end_date - start_date).days + 1
        st.write(f"📅 Obdobie: {total_period_days} dní")

    # Celkový počet dovolenkových žiadostí
    if 'st' in locals() and hasattr(st, 'session_state') and 'vacation_requests' in st.session_state:
        total_vacation_requests = sum(len(requests) for requests in st.session_state.vacation_requests.values())
        st.write(f"🏖️ Dovolenkové žiadosti: {total_vacation_requests}")

    st.markdown("---")
    st.subheader("💡 Tipy na optimalizáciu")
    st.markdown("""
    - Udržujte vyváženú záťaž medzi zamestnancami
    - Používajte preferencie smien pre vyššiu spokojnosť
    - Pravidelne kontrolujte využitie rozpočtu
    - Plánujte dovolenky s dostatočným predstihom
    - Monitorujte víkendové pokrytie
    """)

    st.markdown("---")
    st.caption("© 2025 PlanME Pro - All rights reserved")