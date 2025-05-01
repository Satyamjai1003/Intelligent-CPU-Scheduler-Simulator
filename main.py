import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import time
import copy #import copy lets you make copies of lists, dictionaries, or other objects — especially when you want a completely separate copy.
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from streamlit_extras.metric_cards import style_metric_cards    # For creating the number like boxes.
from streamlit_extras.stylable_container import stylable_container  # adding background colors, borders, shadows, or even rounded corners.
from streamlit_extras.add_vertical_space import add_vertical_space

# =============================================
# Configuration and Constants
# =============================================

# Configure Gemini API (replace with your actual API key)
GEMINI_API_KEY = "AIzaSyC-uUxrwle-H7bC4UsiE1Y9XRW8hx-YUbk"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Algorithm information dictionary
ALGORITHM_INFO = {
    "FCFS": {
        "name": "First-Come First-Served",
        "description": "Processes are executed in the order they arrive. Simple but can lead to long waiting times for short processes behind long ones.",
        "pros": ["Simple to implement", "No starvation"],
        "cons": ["Poor performance for short processes", "Not optimal for average waiting time"],
        "best_for": "Batch systems where simplicity is more important than performance",
        "link": "https://www.geeksforgeeks.org/program-for-fcfs-cpu-scheduling-set-1/",
        "color": "#1f77b4"
    },
    "SJF": {
        "name": "Shortest Job First",
        "description": "The process with the smallest burst time gets executed next. Can be preemptive or non-preemptive.",
        "pros": ["Minimizes average waiting time", "Optimal for turnaround time"],
        "cons": ["Difficult to predict burst times", "Can lead to starvation of long processes"],
        "best_for": "Batch systems where burst times are known in advance",
        "link": "https://www.geeksforgeeks.org/program-for-shortest-job-first-or-sjf-cpu-scheduling-set-1-non-preemptive/",
        "color": "#ff7f0e"
    },
    "Round Robin": {
        "name": "Round Robin",
        "description": "Each process gets a small unit of CPU time (time quantum). Fair and starvation-free.",
        "pros": ["Fair allocation of CPU", "No starvation", "Good for time-sharing systems"],
        "cons": ["Performance depends heavily on time quantum size", "Higher context switching overhead"],
        "best_for": "Interactive/time-sharing systems",
        "link": "https://www.geeksforgeeks.org/program-round-robin-scheduling-set-1/",
        "color": "#2ca02c"
    },
    "Priority": {
        "name": "Priority Scheduling",
        "description": "Processes with higher priority are executed first. Can be preemptive or non-preemptive.",
        "pros": ["Important processes get priority", "Flexible priority assignment"],
        "cons": ["Starvation of low priority processes", "Priority inversion problem"],
        "best_for": "Real-time systems where priority matters",
        "link": "https://www.geeksforgeeks.org/priority-cpu-scheduling-with-different-arrival-time-set-2/",
        "color": "#d62728"
    }
}

# =============================================
# Core Scheduling Logic (unchanged from original)
# =============================================

@dataclass
class Process:
    id: int
    arrival_time: int
    burst_time: int
    priority: int = 0
    waiting_time: int = 0
    turnaround_time: int = 0
    end_time: int = 0
    response_time: int = 0
    remaining_time: int = 0 

    def __post_init__(self):
        self.remaining_time = self.burst_time

class CPUScheduler:
    @staticmethod
    def fcfs_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        for p in processes:
            if current_time < p.arrival_time:
                current_time = p.arrival_time
            start = current_time
            current_time += p.burst_time
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            p.response_time = start - p.arrival_time
            gantt.append((p.id, start, current_time))
        return gantt, processes

    @staticmethod
    def sjf_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        completed = []
        ready_queue = processes.copy()
        
        while ready_queue:
            available = [p for p in ready_queue if p.arrival_time <= current_time]
            if not available:
                current_time += 1
                continue
            current_process = min(available, key=lambda x: x.burst_time)
            start = current_time
            current_time += current_process.burst_time
            current_process.end_time = current_time
            current_process.turnaround_time = current_time - current_process.arrival_time
            current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
            current_process.response_time = start - current_process.arrival_time
            gantt.append((current_process.id, start, current_time))
            completed.append(current_process)
            ready_queue.remove(current_process)
        return gantt, completed

    @staticmethod
    def rr_scheduling(processes: List[Process], quantum: int) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        ready_queue = processes.copy()
        completed = []
        
        while ready_queue:
            if not any(p.arrival_time <= current_time for p in ready_queue):
                current_time += 1
                continue
            current_process = ready_queue.pop(0)
            start = current_time
            execution_time = min(quantum, current_process.remaining_time)
            current_time += execution_time
            current_process.remaining_time -= execution_time
            
            if current_process.response_time == 0 and start >= current_process.arrival_time:
                current_process.response_time = start - current_process.arrival_time
                
            gantt.append((current_process.id, start, current_time))
            
            new_arrivals = [p for p in processes if p.arrival_time <= current_time and p not in ready_queue and p not in completed and p != current_process]
            ready_queue.extend(new_arrivals)
            
            if current_process.remaining_time > 0:
                ready_queue.append(current_process)
            else:
                current_process.end_time = current_time
                current_process.turnaround_time = current_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed.append(current_process)
        return gantt, completed

    @staticmethod
    def priority_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        completed = []
        ready_queue = processes.copy()
        
        while ready_queue:
            available = [p for p in ready_queue if p.arrival_time <= current_time]
            if not available:
                current_time += 1
                continue
            current_process = min(available, key=lambda x: x.priority)
            start = current_time
            current_time += current_process.burst_time
            current_process.end_time = current_time
            current_process.turnaround_time = current_time - current_process.arrival_time
            current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
            current_process.response_time = start - current_process.arrival_time
            gantt.append((current_process.id, start, current_time))
            completed.append(current_process)
            ready_queue.remove(current_process)
        return gantt, completed

# =============================================
# Enhanced Visualization Functions
# =============================================

def plot_gantt_chart(gantt: List[Tuple[int, int, int]], algorithm: str) -> go.Figure:
    """Enhanced Gantt chart with start and completion times"""
    if not gantt:
        fig = go.Figure()
        fig.add_annotation(text="No processes scheduled", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                         plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    # Create DataFrame with process information
    df = pd.DataFrame(gantt, columns=['Process', 'Start', 'Finish'])
    df['Duration'] = df['Finish'] - df['Start']
    df['Process'] = 'P' + df['Process'].astype(str)
    
    # Color palette for processes
    colors = px.colors.qualitative.Pastel
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for each process with unique colors
    for i, (process, group) in enumerate(df.groupby('Process')):
        fig.add_trace(go.Bar(
            x=group['Duration'],
            y=group['Process'],
            base=group['Start'],
            orientation='h',
            name=process,
            marker_color=colors[i % len(colors)],
            hoverinfo='text',
            hovertext=[f'Process: {process}<br>Start: {start}<br>Finish: {finish}<br>Duration: {dur}'
                      for start, finish, dur in zip(group['Start'], group['Finish'], group['Duration'])] 
        ))
    
    # Add vertical grid lines
    max_time = df['Finish'].max()
    for t in range(0, max_time + 1):
        fig.add_vline(
            x=t,
            line_width=1,
            line_dash="solid",
            line_color="lightgray",
            opacity=0.5
        )
    
    # Update layout (moved outside the loop)
    fig.update_layout(
        title=f"{algorithm} Scheduling Gantt Chart",
        xaxis=dict(
            title='Time Units',
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[0, max_time + 1],
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linecolor='black',
            ticks="outside"
        ),
        yaxis=dict(
            title='Processes',
            showgrid=False,
            showline=True,
            linecolor='black'
        ),
        # ... rest of the layout remains unchanged
    )
    
    # Add process labels with start and completion times
    for i, row in df.iterrows():
        # Process label in the middle of the bar
        fig.add_annotation(
            x=row['Start'] + row['Duration']/2,
            y=row['Process'],
            text=row['Process'],
            showarrow=False,
            font=dict(color='black', size=12),
            yshift=0
        )
        
        # Start time label at the beginning of the bar
        fig.add_annotation(
            x=row['Start'],
            y=row['Process'],
            text=f"Start: {row['Start']}",
            showarrow=False,
            font=dict(size=10, color='black'),  # Changed to white
            xshift=5,
            yshift=15
        )
        
        # Completion time label at the end of the bar
        fig.add_annotation(
            x=row['Finish'],
            y=row['Process'],
            text=f"End: {row['Finish']}",
            showarrow=False,
            font=dict(size=10, color='black'),  # Changed to white
            xshift=-5,
            yshift=15
        )

    return fig


def plot_performance_metrics(metrics: Dict[str, float]) -> go.Figure:
    """Interactive metrics visualization with better tooltips"""
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    fig = px.bar(df, x='Metric', y='Value', text='Value',
                 color='Metric', color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>"
    )
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="",
        yaxis_title="Value",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    return fig

def plot_algorithm_comparison(comparison_results: Dict[str, Dict[str, float]]) -> go.Figure:
    """Interactive comparison of all algorithms with better tooltips"""
    metrics = ['avg_waiting', 'avg_turnaround', 'avg_response', 'throughput', 'cpu_utilization']
    algorithm_names = list(comparison_results.keys())
    
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=algorithm_names,
            y=[comparison_results[algo][metric] for algo in algorithm_names],
            marker_color=[ALGORITHM_INFO[algo]['color'] for algo in algorithm_names],
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='group',
        title="Algorithm Performance Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Value",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def calculate_metrics(completed: List[Process]) -> Tuple[float, float, float, float, float]:
    if not completed:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    avg_waiting = sum(p.waiting_time for p in completed) / len(completed)
    avg_turnaround = sum(p.turnaround_time for p in completed) / len(completed)
    avg_response = sum(p.response_time for p in completed) / len(completed)
    max_time = max(p.end_time for p in completed)
    throughput = len(completed) / max_time if max_time > 0 else 0.0
    cpu_utilization = (sum(p.burst_time for p in completed) / max_time) * 100 if max_time > 0 else 0.0
    return avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization


# GanttItem class (used in simulation steps for ready queue logic)
class GanttItem:
    def __init__(self, id: str, start: int, end: int):
        self.id = id
        self.start = start
        self.end = end

# SimulationStep class (stores ready queue state)
class SimulationStep:
    def __init__(self, time: int, active_process_id: Optional[str], ready_queue: List[Process],
                 cpu_usage: int, memory_usage: int, disk_io: int, network_usage: int):
        self.time = time
        self.active_process_id = active_process_id
        self.ready_queue = ready_queue
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.disk_io = disk_io
        self.network_usage = network_usage

# Helper function for random resource usage (used in SimulationStep)
def get_random_value(min_value: int, max_value: int) -> int:
    import random
    return random.randint(min_value, max_value)

# Ready queue logic in CPUScheduler methods
class CPUScheduler:
    @staticmethod
    def fcfs_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        for p in processes:
            if current_time < p.arrival_time:
                current_time = p.arrival_time
            start = current_time
            current_time += p.burst_time
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            p.response_time = start - p.arrival_time
            gantt.append((p.id, start, current_time))
        return gantt, processes

    @staticmethod
    def sjf_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        completed = []
        ready_queue = processes.copy()
        
        while ready_queue:
            available = [p for p in ready_queue if p.arrival_time <= current_time]
            if not available:
                current_time += 1
                continue
            current_process = min(available, key=lambda x: x.burst_time)
            start = current_time
            current_time += current_process.burst_time
            current_process.end_time = current_time
            current_process.turnaround_time = current_time - current_process.arrival_time
            current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
            current_process.response_time = start - current_process.arrival_time
            gantt.append((current_process.id, start, current_time))
            completed.append(current_process)
            ready_queue.remove(current_process)
        return gantt, completed

    @staticmethod
    def rr_scheduling(processes: List[Process], quantum: int) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        ready_queue = processes.copy()
        completed = []
        
        while ready_queue:
            if not any(p.arrival_time <= current_time for p in ready_queue):
                current_time += 1
                continue
            current_process = ready_queue.pop(0)
            start = current_time
            execution_time = min(quantum, current_process.remaining_time)
            current_time += execution_time
            current_process.remaining_time -= execution_time
            
            if current_process.response_time == 0 and start >= current_process.arrival_time:
                current_process.response_time = start - current_process.arrival_time
                
            gantt.append((current_process.id, start, current_time))
            
            new_arrivals = [p for p in processes if p.arrival_time <= current_time and p not in ready_queue and p not in completed and p != current_process]
            ready_queue.extend(new_arrivals)
            
            if current_process.remaining_time > 0:
                ready_queue.append(current_process)
            else:
                current_process.end_time = current_time
                current_process.turnaround_time = current_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed.append(current_process)
        return gantt, completed

    @staticmethod
    def priority_scheduling(processes: List[Process]) -> Tuple[List[Tuple[int, int, int]], List[Process]]:
        processes.sort(key=lambda x: x.arrival_time)
        current_time = 0
        gantt = []
        completed = []
        ready_queue = processes.copy()
        
        while ready_queue:
            available = [p for p in ready_queue if p.arrival_time <= current_time]
            if not available:
                current_time += 1
                continue
            current_process = min(available, key=lambda x: x.priority)
            start = current_time
            current_time += current_process.burst_time
            current_process.end_time = current_time
            current_process.turnaround_time = current_time - current_process.arrival_time
            current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
            current_process.response_time = start - current_process.arrival_time
            gantt.append((current_process.id, start, current_time))
            completed.append(current_process)
            ready_queue.remove(current_process)
        return gantt, completed

# Ready queue logic in simulation steps
def generate_simulation_steps(gantt_chart: List[GanttItem], processes: List[Process]) -> 'SimulationData':
    class SimulationData:
        def __init__(self, steps: List[SimulationStep], total_time: int, idle_time: int, processing_time: int,
                     cpu_utilization: float):
            self.steps = steps
            self.total_time = total_time
            self.idle_time = idle_time
            self.processing_time = processing_time
            self.cpu_utilization = cpu_utilization

    if not gantt_chart:
        return SimulationData([], 0, 0, 0, 0)

    total_time = max(item.end for item in gantt_chart)
    steps = []

    for time in range(total_time + 1):
        active_gantt_item = next((item for item in gantt_chart if item.start <= time < item.end), None)
        active_process_id = active_gantt_item.id if active_gantt_item else None

        # Ready queue logic: Include processes that have arrived and are not currently executing
        ready_queue = sorted(
            [p for p in processes if p.arrival_time <= time and not any(item.id == p.id and item.start <= time < item.end for item in gantt_chart)],
            key=lambda p: p.arrival_time
        )

        # Generate simulated resource usage (included for completeness in SimulationStep)
        cpu_usage = get_random_value(60, 95) if active_process_id else get_random_value(5, 20)
        memory_usage = get_random_value(40, 90)
        disk_io = get_random_value(5, 50)
        network_usage = get_random_value(1, 30)

        steps.append(SimulationStep(time, active_process_id, ready_queue, cpu_usage, memory_usage, disk_io, network_usage))

    idle_steps = sum(1 for step in steps if step.active_process_id is None)
    idle_time = (idle_steps / (total_time + 1)) * 100
    processing_time = total_time - idle_steps
    cpu_utilization = 100 - idle_time

    return SimulationData(steps, total_time, idle_time, processing_time, cpu_utilization)

# Visualization of the ready queue
def plot_process_queue(ready_queue, current_process=None):
    fig = go.Figure()
    for i, process in enumerate(ready_queue):
        color = "#0ea5e9" if process.id == (current_process.id if current_process else None) else "#808080"
        fig.add_annotation(
            x=i + 1,
            y=0.5,
            text=process.id,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor=color,
            borderpad=10,
            bordercolor="none",
            borderwidth=0
        )
    fig.update_layout(
        title="Process Queue",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        template="plotly_white",
        height=200
    )
    if not ready_queue:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No processes in queue",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
    return fig

# Display ready queue in Streamlit
def display_ready_queue(queue: List[Process], current_process: Optional[Process] = None):
    # Initialize executed processes list in session state if not present
    if "executed_processes" not in st.session_state:
        st.session_state.executed_processes = []

    # Add current process to executed processes if it exists and isn't already added
    if current_process and current_process.id not in st.session_state.executed_processes:
        st.session_state.executed_processes.append(current_process.id)

    
        st.markdown("### Ready Queue Status")
        
        # Display CPU section with execution sequence
        if st.session_state.executed_processes:
            execution_sequence = " -> ".join(f"P{id}" for id in st.session_state.executed_processes)
            st.markdown(f"**CPU:** `{execution_sequence}`")
        else:
            st.markdown("**CPU:** `Idle`")
        
        st.markdown("---")
        st.markdown("**Waiting:**")
        if queue:
            for process in queue:
                st.code(f"P{process.id}")
        else:
            st.markdown("No processes in queue")

# Ready queue in real-time simulation
def real_time_simulation(processes: List[Process], algorithm: str, quantum: int):
    scheduler = CPUScheduler()
    algo_map = {
        "FCFS": scheduler.fcfs_scheduling,
        "SJF": scheduler.sjf_scheduling,
        "Round Robin": lambda p: scheduler.rr_scheduling(p, quantum),
        "Priority": scheduler.priority_scheduling
    }
    
    # Reset executed processes list at the start of simulation
    st.session_state.executed_processes = []
    
    processes_copy = copy.deepcopy(processes)
    current_time = 0
    ready_queue = processes_copy.copy()
    completed = []
    gantt = []
    
    placeholder = st.empty()
    
    while ready_queue or len(completed) < len(processes):
        with placeholder.container():
            available = [p for p in ready_queue if p.arrival_time <= current_time]
            if not available and ready_queue:
                current_time += 1
                display_ready_queue(ready_queue)
                st.write(f"Time: {current_time} - Idle")
                time.sleep(0.5)
                continue
                
            if algorithm == "Round Robin":
                if not available:
                    current_time += 1
                    continue
                current_process = ready_queue.pop(0)
                start = current_time
                execution_time = min(quantum, current_process.remaining_time)
                current_process.remaining_time -= execution_time
                current_time += execution_time
                gantt.append((current_process.id, start, current_time))
                display_ready_queue(ready_queue, current_process)
                st.write(f"Time: {current_time} - Executing P{current_process.id}")
                
                if current_process.remaining_time > 0:
                    ready_queue.append(current_process)
                else:
                    current_process.end_time = current_time
                    current_process.turnaround_time = current_time - current_process.arrival_time
                    current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                    if current_process.response_time == 0:
                        current_process.response_time = start - current_process.arrival_time
                    completed.append(current_process)
            else:
                gantt_segment, completed_segment = algo_map[algorithm](available[:1])
                if not gantt_segment:
                    current_time += 1
                    continue
                current_process = completed_segment[0]
                ready_queue.remove(current_process)
                completed.append(current_process)
                gantt.extend(gantt_segment)
                display_ready_queue(ready_queue, current_process)
                st.write(f"Time: {current_time} - Executing P{current_process.id}")
                current_time = current_process.end_time
                
            time.sleep(0.5)
    
    return gantt, completed



# =============================================
# AI Recommendation Function
# =============================================
def get_ai_recommendation(processes: List[Process], current_algorithm: str) -> str:
    """
    Get intelligent algorithm recommendation based on process characteristics
    Uses either Gemini API (if configured) or a built-in rules-based fallback
    """
    try:
        # First try using Gemini API if configured
        if 'GEMINI_API_KEY' in globals() and GEMINI_API_KEY != "YOUR_API_KEY":
            return get_gemini_recommendation(processes, current_algorithm)
    except Exception as e:
        st.warning(f"Gemini API unavailable: {str(e)}. Using built-in recommendation system.")
    
    # Fallback to rules-based recommendation
    return get_fallback_recommendation(processes, current_algorithm)

def get_gemini_recommendation(processes: List[Process], current_algorithm: str) -> str:
    """Get recommendation using Gemini API"""
    try:
        # Initialize the model with the correct API version
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')  # Use the correct model name
        
        prompt = f"""
        Analyze these processes for CPU scheduling:
        {[f'P{p.id}(arrival:{p.arrival_time}, burst:{p.burst_time}, priority:{p.priority})' for p in processes]}
        
        Current algorithm: {current_algorithm}
        Recommend the best algorithm (FCFS, SJF, Round Robin, Priority) and explain why in 50 words.
        Mention one potential drawback.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

def get_fallback_recommendation(processes: List[Process], current_algorithm: str) -> str:
    """Intelligent fallback recommendation system"""
    # Calculate process characteristics
    burst_times = [p.burst_time for p in processes]
    priorities = [p.priority for p in processes]
    arrival_times = [p.arrival_time for p in processes]
    
    avg_burst = sum(burst_times) / len(burst_times)
    burst_variation = max(burst_times) - min(burst_times)
    priority_variation = max(priorities) - min(priorities) if any(p > 0 for p in priorities) else 0
    arrival_variation = max(arrival_times) - min(arrival_times)
    
    # Decision logic
    if priority_variation > 3 and current_algorithm != "Priority":
        return (
            "Recommendation: Priority Scheduling\n"
            "Why: Significant priority variation between processes detected\n"
            "Drawback: May starve low-priority processes"
        )
    elif burst_variation > 5 and avg_burst < 10 and current_algorithm != "SJF":
        return (
            "Recommendation: SJF (Shortest Job First)\n"
            "Why: Large variation in short burst times favors SJF\n"
            "Drawback: Requires accurate burst time estimates"
        )
    elif arrival_variation > 5 and current_algorithm != "FCFS":
        return (
            "Recommendation: FCFS (First-Come First-Served)\n"
            "Why: Significant arrival time variation works well with FCFS\n"
            "Drawback: May have poor turnaround for short processes"
        )
    else:
        return (
            "Recommendation: Round Robin\n"
            "Why: Balanced approach for mixed workloads\n"
            "Drawback: Higher context switching overhead"
        )
    
# =============================================
# UI Components
# =============================================

def create_algorithm_info(algorithm: str):
    """Display detailed information about the selected algorithm"""
    info = ALGORITHM_INFO[algorithm]
    
    with st.expander(f"ℹ️ About {info['name']}", expanded=True):
        st.markdown(f"**Description:** {info['description']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pros:**")
            for pro in info['pros']:
                st.markdown(f"- ✅ {pro}")
        
        with col2:
            st.markdown("**Cons:**")
            for con in info['cons']:
                st.markdown(f"- ❌ {con}")
        
        st.markdown(f"**Best for:** {info['best_for']}")
        st.markdown(f"🔗 [Learn more about {info['name']}]({info['link']})")

def create_process_input_form(algorithm: str):
    """Styled process input form with validation"""
    with st.expander("➕ Add New Process", expanded=True):
        with st.form(key='process_input_form'):
            cols = st.columns([1,1,1,1,2])
            
            with cols[0]:
                id = st.number_input("Process ID", min_value=0, step=1, key="id")
            with cols[1]:
                arrival_time = st.number_input("Arrival Time", min_value=0, step=1, key="arrival")
            with cols[2]:
                burst_time = st.number_input("Burst Time", min_value=1, step=1, key="burst")
            with cols[3]:
                if algorithm == "Priority":
                    priority = st.number_input("Priority", min_value=0, step=1, key="priority", value=0)
                else:
                    priority = 0
                    st.write("Priority")  # Placeholder
                    st.write("N/A")
            
            with cols[4]:
                st.write("")  # Spacer
                st.write("")  # Spacer
                submit_button = st.form_submit_button("Add Process", use_container_width=True)
            
            if submit_button:
                # Validation
                if id in [p.id for p in st.session_state.processes]:
                    st.error(f"Process ID {id} already exists!")
                elif arrival_time < 0:
                    st.error("Arrival time cannot be negative!")
                elif burst_time <= 0:
                    st.error("Burst time must be positive!")
                else:
                    new_process = Process(id, arrival_time, burst_time, priority)
                    st.session_state.processes.append(new_process)
                    st.session_state.new_process = id
                    st.success(f"Process P{id} added successfully!")
                    st.rerun()

def display_process_table():
    """Styled process table with interactive controls"""
    if not st.session_state.processes:
        st.info("No processes added yet. Add processes using the form above.")
        return
    
    with st.expander("📋 Current Process Table", expanded=True):
        process_data = {
            "ID": [f"P{p.id}" for p in st.session_state.processes],
            "Arrival": [p.arrival_time for p in st.session_state.processes],
            "Burst": [p.burst_time for p in st.session_state.processes],
            "Priority": [p.priority for p in st.session_state.processes] if "Priority" in st.session_state.algorithm else ["N/A"]*len(st.session_state.processes)
        }
        
        df = pd.DataFrame(process_data)
        
        # Apply row highlighting for new process
        def highlight_row(row):
            if st.session_state.new_process == int(row['ID'][1:]):
                return ['background-color: #e6f7ff' if st.session_state.theme == "Light" else 'background-color: #FFFFFF'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Clear All Processes", use_container_width=True):
                st.session_state.processes = []
                st.session_state.new_process = None
                st.success("All processes cleared!")
                st.rerun()
        with col2:
            if st.button("Generate Random Processes", use_container_width=True):
                generate_random_processes()
                st.rerun()
        with col3:
            if st.button("Load Example", use_container_width=True):
                load_example_processes()
                st.rerun()

def generate_random_processes():
    """Generate 5 random processes"""
    import random
    st.session_state.processes = []
    for i in range(5):
        arrival = random.randint(0, 5)
        burst = random.randint(1, 10)
        priority = random.randint(1, 5) if st.session_state.algorithm == "Priority" else 0
        st.session_state.processes.append(Process(i+1, arrival, burst, priority))
    st.session_state.new_process = None

def load_example_processes():
    """Load example processes based on selected algorithm"""
    examples = {
        "FCFS": [
            Process(1, 0, 5, 0),
            Process(2, 1, 3, 0),
            Process(3, 2, 8, 0),
            Process(4, 3, 6, 0)
        ],
        "SJF": [
            Process(1, 0, 6, 0),
            Process(2, 2, 8, 0),
            Process(3, 4, 7, 0),
            Process(4, 5, 3, 0)
        ],
        "Round Robin": [
            Process(1, 0, 5, 0),
            Process(2, 1, 3, 0),
            Process(3, 2, 6, 0),
            Process(4, 4, 4, 0)
        ],
        "Priority": [
            Process(1, 0, 5, 1),
            Process(2, 1, 3, 3),
            Process(3, 2, 8, 2),
            Process(4, 3, 6, 4)
        ]
    }
    st.session_state.processes = examples.get(st.session_state.algorithm, [])
    st.session_state.new_process = None

def display_metrics(avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization):
    """Styled metrics display with cards"""
    with st.expander("📊 Performance Metrics", expanded=True):
        cols = st.columns(5)
        with cols[0]:
            st.metric("Avg Waiting", f"{avg_waiting:.2f}", help="Average time processes spend waiting in ready queue")
        with cols[1]:
            st.metric("Avg Turnaround", f"{avg_turnaround:.2f}", help="Average time from arrival to completion")
        with cols[2]:
            st.metric("Avg Response", f"{avg_response:.2f}", help="Average time from arrival to first response")
        with cols[3]:
            st.metric("Throughput", f"{throughput:.3f}", help="Processes completed per unit time")
        with cols[4]:
            st.metric("CPU Utilization", f"{cpu_utilization:.1f}%", help="Percentage of CPU time used")
        
        # Apply styling to metrics
        style_metric_cards(border_left_color="#1f77b4", border_color="#1f77b4")


# =============================================
# Algorithm Comparison Functions
# =============================================

def compare_algorithms(processes: List[Process], quantum: int = 2) -> Dict[str, Dict[str, float]]:
    """Compare all algorithms and return their metrics"""
    scheduler = CPUScheduler()
    algorithms = {
        "FCFS": lambda p: scheduler.fcfs_scheduling(p),
        "SJF": lambda p: scheduler.sjf_scheduling(p),
        "Round Robin": lambda p: scheduler.rr_scheduling(p, quantum),
        "Priority": lambda p: scheduler.priority_scheduling(p)
    }
    
    results = {}
    for name, algo in algorithms.items():
        processes_copy = copy.deepcopy(processes)
        _, completed = algo(processes_copy)
        avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization = calculate_metrics(completed)
        results[name] = {
            "avg_waiting": avg_waiting,
            "avg_turnaround": avg_turnaround,
            "avg_response": avg_response,
            "throughput": throughput,
            "cpu_utilization": cpu_utilization
        }
    return results

def get_algorithm_suggestion(results: Dict[str, Dict[str, float]]) -> str:
    """Analyze metrics and return the best algorithm suggestion"""
    if not results:
        return "No algorithms to compare"
    
    # Weights for different metrics
    weights = {
        'avg_waiting': 0.3,
        'avg_turnaround': 0.3,
        'avg_response': 0.2,
        'throughput': 0.1,
        'cpu_utilization': 0.1
    }
    
    algorithms = list(results.keys())
    metrics = list(results[algorithms[0]].keys())
    
    # Normalize metrics (higher is better)
    normalized = {}
    for metric in metrics:
        values = [results[algo][metric] for algo in algorithms]
        if metric in ['avg_waiting', 'avg_turnaround', 'avg_response']:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized[metric] = [1.0] * len(values)
            else:
                normalized[metric] = [(max_val - v)/(max_val - min_val) for v in values]
        else:
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                normalized[metric] = [1.0] * len(values)
            else:
                normalized[metric] = [(v - min_val)/(max_val - min_val) for v in values]
    
    # Calculate weighted scores
    scores = []
    for i in range(len(algorithms)):
        score = sum(normalized[metric][i] * weights[metric] for metric in metrics)
        scores.append(score)
    
    # Get best algorithm
    best_idx = scores.index(max(scores))
    best_algo = algorithms[best_idx]
    best_metrics = results[best_algo]
    
    explanations = []
    if best_metrics['avg_waiting'] == min(results[algo]['avg_waiting'] for algo in algorithms):
        explanations.append("lowest average waiting time")
    if best_metrics['avg_turnaround'] == min(results[algo]['avg_turnaround'] for algo in algorithms):
        explanations.append("lowest average turnaround time")
    if best_metrics['avg_response'] == min(results[algo]['avg_response'] for algo in algorithms):
        explanations.append("lowest average response time")
    if best_metrics['throughput'] == max(results[algo]['throughput'] for algo in algorithms):
        explanations.append("highest throughput")
    if best_metrics['cpu_utilization'] == max(results[algo]['cpu_utilization'] for algo in algorithms):
        explanations.append("best CPU utilization")
    
    return f"**{best_algo}** - Recommended because it has {', '.join(explanations)}."

# =============================================
# Main Application
# =============================================

def main():
    # Set page configuration
    st.set_page_config(
        page_title="CPU Scheduler Simulator",
        page_icon="⏱️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "processes" not in st.session_state:
        st.session_state.processes = []
    if "new_process" not in st.session_state:
        st.session_state.new_process = None
    if "executed_processes" not in st.session_state:
        st.session_state.executed_processes = []
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "algorithm" not in st.session_state:
        st.session_state.algorithm = "FCFS"
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⏱️ CPU Scheduler")
        st.markdown("""
            <style>
                .stSelectbox div {
                cursor: pointer !important;
                }
            </style>
            """, unsafe_allow_html=True)
        # Theme selector
        st.session_state.theme = st.selectbox(
            "Theme",
            ["light", "dark"],
            index=0,
            key="theme_selector",
            help="Change the color scheme of the application"
        )
        
        # Apply theme
        if st.session_state.theme == "dark":
            plt.style.use('dark_background')
            st.markdown("""
            <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: white;
                }
                .sidebar .sidebar-content {
                    background-color: #2E2E2E;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #4FC3F7;
                }
                .stSelectbox div {
                    cursor: pointer !important;
                }       
            </style>
            """, unsafe_allow_html=True)
        else:
            plt.style.use('default')
            st.markdown("""
            <style>
                .stApp {
                    background-color: #f5f5f5;
                }
                .sidebar .sidebar-content {
                    background-color: #ffffff;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #1f77b4;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Algorithm selection
        st.session_state.algorithm = st.selectbox(
            "Scheduling Algorithm",
            list(ALGORITHM_INFO.keys()),
            index=0,
            key="algorithm_selector",
            help="Select the CPU scheduling algorithm to simulate"
        )
        
        # Display algorithm info
        create_algorithm_info(st.session_state.algorithm)
        
        # Simulation mode
        st.session_state.simulation_mode = st.radio(
            "Simulation Mode",
            ["Standard", "Real-Time"],
            index=0,
            key="mode_selector",
            help="Standard shows results immediately, Real-Time shows step-by-step execution"
        )
        
        # Time quantum for Round Robin
        if st.session_state.algorithm == "Round Robin":
            st.session_state.quantum = st.slider(
                "Time Quantum",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                key="quantum_selector",
                help="Time slice for Round Robin scheduling"
            )
        else:
            st.session_state.quantum = 1
            
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("""
        This interactive simulator demonstrates various CPU scheduling algorithms.
        Add processes and run simulations to see how different algorithms perform.
        """)
    
    # Main content
    st.title("CPU Scheduling Algorithm Simulator")
    st.markdown("Visualize and compare different CPU scheduling algorithms with this interactive tool.")
    
    # Process input section
    create_process_input_form(st.session_state.algorithm)
    display_process_table()
    
    # Run simulation button
    if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
        if not st.session_state.processes:
            st.warning("Please add at least one process to run the simulation")
        else:
            run_simulation()
    
    # Add some space before the footer
    add_vertical_space(3)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>CPU Scheduler Simulator | Created with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def run_simulation():
    """Run the selected simulation"""
    processes = st.session_state.processes
    algorithm = st.session_state.algorithm
    quantum = st.session_state.quantum
    simulation_mode = st.session_state.simulation_mode
    
    scheduler = CPUScheduler()
    algo_map = {
        "FCFS": scheduler.fcfs_scheduling,
        "SJF": scheduler.sjf_scheduling,
        "Round Robin": lambda p: scheduler.rr_scheduling(p, quantum),
        "Priority": scheduler.priority_scheduling
    }
    try:
        if simulation_mode == "Real-Time":
            st.subheader("🎬 Real-Time Simulation")
            with st.spinner("Running real-time simulation..."):
                processes_copy = copy.deepcopy(processes)
                gantt, completed = real_time_simulation(processes_copy, algorithm, quantum)
                # gantt, completed = algo_map[algorithm](processes_copy)  
                # gantt, completed = algo_map[algorithm](processes_copy)
                
                # Display Gantt chart
                st.plotly_chart(plot_gantt_chart(gantt, algorithm), use_container_width=True)
                
                # Display simulation steps (optional, for resource usage)
                simulation_data = generate_simulation_steps(
                    [GanttItem(id, start, end) for id, start, end in gantt],
                    processes_copy
                )
                st.write("Resource Usage Over Time:")
                resource_df = pd.DataFrame([
                    {
                        "Time": step.time,
                        "Active Process": f"P{step.active_process_id}" if step.active_process_id else "Idle",
                        "Ready Queue": ", ".join(f"P{p.id}" for p in step.ready_queue),
                        "CPU Usage (%)": step.cpu_usage,
                        "Memory Usage (%)": step.memory_usage,
                        "Disk I/O": step.disk_io,
                        "Network Usage": step.network_usage
                    }
                    for step in simulation_data.steps
                ])
                st.dataframe(resource_df, use_container_width=True, hide_index=True)
                
                # Display metrics
                avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization = calculate_metrics(completed)
                display_metrics(avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization)
        else:
            st.subheader("📊 Simulation Results")
            processes_copy = copy.deepcopy(processes)
            gantt, completed = algo_map[algorithm](processes_copy)
            
            # Standard simulation results
            with st.spinner("Calculating results..."):
                time.sleep(1)  # Simulate processing time
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Gantt Chart", "Metrics", "Details"])
                
                with tab1:
                    st.plotly_chart(plot_gantt_chart(gantt, algorithm), use_container_width=True)
                
                with tab2:
                    avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization = calculate_metrics(completed)
                    display_metrics(avg_waiting, avg_turnaround, avg_response, throughput, cpu_utilization)
                    st.plotly_chart(plot_performance_metrics({
                        "Avg Waiting": avg_waiting,
                        "Avg Turnaround": avg_turnaround,
                        "Avg Response": avg_response,
                        "Throughput": throughput,
                        "CPU Utilization": cpu_utilization
                    }), use_container_width=True)
                
                with tab3:
                    st.dataframe(pd.DataFrame({
                        "Process": [f"P{p.id}" for p in completed],
                        "Arrival": [p.arrival_time for p in completed],
                        "Burst": [p.burst_time for p in completed],
                        "Waiting": [p.waiting_time for p in completed],
                        "Turnaround": [p.turnaround_time for p in completed],
                        "Response": [p.response_time for p in completed],
                        "End Time": [p.end_time for p in completed]
                    }), use_container_width=True, hide_index=True)
        
        # Algorithm comparison section
        st.subheader("🔍 Algorithm Comparison")
        with st.spinner("Comparing all scheduling algorithms..."):
            comparison_results = compare_algorithms(processes, quantum)
            
            # Display comparison visualization
            st.plotly_chart(plot_algorithm_comparison(comparison_results), use_container_width=True)
            
            # Display suggestion
            st.success(get_algorithm_suggestion(comparison_results))
            
            # Display AI recommendation
            recommendation = get_ai_recommendation(processes, algorithm)
            st.info("🤖 AI Recommendation")
            st.markdown(f"```\n{recommendation}\n```")
    
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")

if __name__ == "__main__":
    main()