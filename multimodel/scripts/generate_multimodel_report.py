#!/usr/bin/env python3
"""Generate HTML report for multimodel pipeline results."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import html


def load_sample_data(sample_dir: Path) -> Dict[str, Any]:
    base_path = sample_dir / "base.json"
    if not base_path.exists():
        return None
    with open(base_path) as f:
        return json.load(f)


def get_color_for_drop(drop: float) -> str:
    if drop >= 0.5:
        return "#ffcdd2"
    elif drop >= 0.2:
        return "#ffe0b2"
    elif drop >= 0.1:
        return "#fff9c4"
    else:
        return "#c8e6c9"


def render_sentence_html(idx: int, total: int, anchor: Dict[str, Any]) -> str:
    drop = anchor.get("accuracy_drop", 0)
    is_anchor = anchor.get("is_anchor", False)
    acc_with = anchor.get("accuracy_with_sentence", 0) * 100
    acc_without = anchor.get("accuracy_without_sentence", 0) * 100
    text = html.escape(anchor.get("sentence_text", ""))
    
    bg_color = get_color_for_drop(drop) if is_anchor else "#f5f5f5"
    border_color = "#f44336" if drop >= 0.5 else "#ff9800" if is_anchor else "#ddd"
    drop_class = "bad" if drop >= 0.1 else "good"
    
    return f'''
    <div class="sentence" style="background: {bg_color}; border-left-color: {border_color};">
        <div class="sentence-header">
            <span class="sentence-idx">#{idx + 1}/{total}</span>
            <div class="sentence-stats">
                <span class="stat" title="Accuracy with this sentence">With: {acc_with:.0f}%</span>
                <span class="stat" title="Accuracy without this sentence">Without: {acc_without:.0f}%</span>
                <span class="stat {drop_class}" title="Accuracy drop when removed">Drop: {drop * 100:.1f}%</span>
                {"<span class='stat anchor-tag'>ANCHOR</span>" if is_anchor else ""}
            </div>
        </div>
        <div class="sentence-text">{text}</div>
    </div>
    '''


def render_trajectory_chart_js(trajectory: List[Dict]) -> str:
    if not trajectory:
        return ""
    
    labels = [f"S{i+1}" for i in range(len(trajectory))]
    correct_probs = [t.get("prob_correct", 0) * 100 for t in trajectory]
    wrong_probs = [t.get("prob_wrong", 0) * 100 for t in trajectory]
    ratios = [t.get("prob_ratio", 0) for t in trajectory]
    
    return f'''
    const trajLabels = {json.dumps(labels)};
    const correctProbs = {json.dumps(correct_probs)};
    const wrongProbs = {json.dumps(wrong_probs)};
    const probRatios = {json.dumps(ratios)};
    
    new Chart(document.getElementById('probChart'), {{
        type: 'line',
        data: {{
            labels: trajLabels,
            datasets: [{{
                label: 'Correct Answer',
                data: correctProbs,
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                fill: true,
                tension: 0.3
            }}, {{
                label: 'Wrong Answer',
                data: wrongProbs,
                borderColor: '#f44336',
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                fill: true,
                tension: 0.3
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                y: {{ min: 0, max: 100, title: {{ display: true, text: 'Probability (%)' }} }}
            }},
            plugins: {{
                title: {{ display: true, text: 'Answer Probabilities Over CoT' }}
            }}
        }}
    }});
    
    new Chart(document.getElementById('ratioChart'), {{
        type: 'line',
        data: {{
            labels: trajLabels,
            datasets: [{{
                label: 'Log Prob Ratio (correct/wrong)',
                data: probRatios,
                borderColor: '#2196F3',
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                fill: true,
                tension: 0.3
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                y: {{ title: {{ display: true, text: 'Log Ratio' }} }}
            }},
            plugins: {{
                title: {{ display: true, text: 'Probability Ratio Over CoT' }},
                annotation: {{
                    annotations: {{
                        line1: {{ type: 'line', yMin: 0, yMax: 0, borderColor: '#666', borderDash: [5,5] }}
                    }}
                }}
            }}
        }}
    }});
    '''


def render_sample_html(sample_id: str, data: Dict[str, Any], is_sycophantic: bool) -> str:
    question = html.escape(data.get("question", ""))
    choices = data.get("choices", [])
    correct_idx = data.get("correct_idx", 0)
    distractor_idx = data.get("distractor_idx", 0)
    cot = html.escape(data.get("cot", "")[:2000])
    response = html.escape(data.get("response", "")[-500:] if data.get("response") else "")
    
    anchor_analysis = data.get("anchor_analysis", [])
    trajectory = data.get("trajectory", [])
    
    result_class = "incorrect" if is_sycophantic else "correct"
    result_text = "SYCOPHANTIC" if is_sycophantic else "CORRECT"
    
    n_anchors = sum(1 for a in anchor_analysis if a.get("is_anchor", False))
    high_impact = sum(1 for a in anchor_analysis if a.get("accuracy_drop", 0) >= 0.3)
    
    choices_html = ""
    for i, choice in enumerate(choices):
        choice_class = ""
        if i == correct_idx:
            choice_class = "correct-choice"
        elif i == distractor_idx:
            choice_class = "distractor-choice"
        choices_html += f'<li class="{choice_class}">{html.escape(str(choice))}</li>'
    
    sentences_html = ""
    for i, anchor in enumerate(anchor_analysis):
        sentences_html += render_sentence_html(i, len(anchor_analysis), anchor)
    
    chart_js = render_trajectory_chart_js(trajectory)
    
    return f'''
    <div class="sample-card">
        <div class="sample-header">
            <h2>{sample_id}</h2>
            <span class="result-badge {result_class}">{result_text}</span>
        </div>
        
        <div class="metadata">
            <div class="metadata-card">
                <h3>Total Sentences</h3>
                <div class="value">{len(anchor_analysis)}</div>
            </div>
            <div class="metadata-card">
                <h3>Anchors Found</h3>
                <div class="value">{n_anchors}</div>
            </div>
            <div class="metadata-card">
                <h3>High Impact (≥30%)</h3>
                <div class="value">{high_impact}</div>
            </div>
        </div>
        
        <div class="section">
            <h3>Question</h3>
            <div class="question-box">{question}</div>
            <ul class="choices">{choices_html}</ul>
        </div>
        
        <div class="section">
            <h3>Probability Trajectory</h3>
            <div class="tabs">
                <button class="tab active" onclick="showTab('{sample_id}_probs', this)">Probabilities</button>
                <button class="tab" onclick="showTab('{sample_id}_ratio', this)">Ratio</button>
            </div>
            <div id="{sample_id}_probs" class="tab-content active">
                <div class="chart-container">
                    <canvas id="probChart_{sample_id}"></canvas>
                </div>
            </div>
            <div id="{sample_id}_ratio" class="tab-content">
                <div class="chart-container">
                    <canvas id="ratioChart_{sample_id}"></canvas>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Chain-of-Thought Sentences</h3>
            <div class="sentences-container">
                {sentences_html}
            </div>
        </div>
        
        <div class="section">
            <h3>Final Answer (last 500 chars)</h3>
            <div class="final-answer">{response}</div>
        </div>
    </div>
    
    <script>
    (function() {{
        const trajLabels = {json.dumps([f"S{i+1}" for i in range(len(trajectory))])};
        const correctProbs = {json.dumps([t.get("prob_correct", 0) * 100 for t in trajectory])};
        const wrongProbs = {json.dumps([t.get("prob_wrong", 0) * 100 for t in trajectory])};
        const probRatios = {json.dumps([t.get("prob_ratio", 0) for t in trajectory])};
        
        new Chart(document.getElementById('probChart_{sample_id}'), {{
            type: 'line',
            data: {{
                labels: trajLabels,
                datasets: [{{
                    label: 'Correct Answer',
                    data: correctProbs,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.3
                }}, {{
                    label: 'Wrong Answer',
                    data: wrongProbs,
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ min: 0, max: 100, title: {{ display: true, text: 'Probability (%)' }} }}
                }}
            }}
        }});
        
        new Chart(document.getElementById('ratioChart_{sample_id}'), {{
            type: 'line',
            data: {{
                labels: trajLabels,
                datasets: [{{
                    label: 'Log Prob Ratio',
                    data: probRatios,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    annotation: {{
                        annotations: {{
                            line1: {{ type: 'line', yMin: 0, yMax: 0, borderColor: '#666', borderDash: [5,5] }}
                        }}
                    }}
                }}
            }}
        }});
    }})();
    </script>
    '''


def generate_report(output_dir: Path, model_name: str, n_samples: int = 10) -> Path:
    base_data_dir = output_dir / model_name / "base_data"
    
    sample_dirs = sorted([d for d in base_data_dir.iterdir() if d.is_dir()])
    
    syco_samples = []
    non_syco_samples = []
    
    for sample_dir in sample_dirs:
        data = load_sample_data(sample_dir)
        if data is None:
            continue
        
        is_syco = data.get("is_sycophantic", False)
        has_anchors = any(a.get("is_anchor", False) for a in data.get("anchor_analysis", []))
        
        if has_anchors:
            if is_syco and len(syco_samples) < n_samples:
                syco_samples.append((sample_dir.name, data))
            elif not is_syco and len(non_syco_samples) < n_samples:
                non_syco_samples.append((sample_dir.name, data))
        
        if len(syco_samples) >= n_samples and len(non_syco_samples) >= n_samples:
            break
    
    samples_html = ""
    
    samples_html += "<h2 class='section-title syco'>Sycophantic Samples</h2>"
    for sample_id, data in syco_samples:
        samples_html += render_sample_html(sample_id, data, is_sycophantic=True)
    
    samples_html += "<h2 class='section-title non-syco'>Non-Sycophantic Samples</h2>"
    for sample_id, data in non_syco_samples:
        samples_html += render_sample_html(sample_id, data, is_sycophantic=False)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodel Report - {model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}
        .section-title {{
            margin-top: 40px;
            padding: 15px;
            border-radius: 8px;
            color: white;
        }}
        .section-title.syco {{ background: #f44336; }}
        .section-title.non-syco {{ background: #4CAF50; }}
        .sample-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
        }}
        .sample-header h2 {{ margin: 0; color: #333; }}
        .result-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .result-badge.incorrect {{ background: #ffcdd2; color: #c62828; }}
        .result-badge.correct {{ background: #c8e6c9; color: #2e7d32; }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .metadata-card {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metadata-card h3 {{
            margin: 0 0 8px 0;
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        .metadata-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .section h3 {{
            color: #555;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        .question-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .choices {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .choices li {{
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 6px;
            background: #f5f5f5;
        }}
        .choices li.correct-choice {{
            background: #c8e6c9;
            border-left: 4px solid #4CAF50;
        }}
        .choices li.distractor-choice {{
            background: #ffcdd2;
            border-left: 4px solid #f44336;
        }}
        .chart-container {{
            height: 250px;
            margin-bottom: 15px;
        }}
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        .tab {{
            padding: 8px 16px;
            background: #e0e0e0;
            border: none;
            border-radius: 4px 4px 0 0;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .tab.active {{
            background: white;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .sentences-container {{
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 15px;
        }}
        .sentence {{
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }}
        .sentence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .sentence-idx {{
            font-weight: bold;
            color: #666;
            font-size: 0.85em;
        }}
        .sentence-stats {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .stat {{
            padding: 3px 8px;
            border-radius: 4px;
            background: #f0f0f0;
            font-size: 0.8em;
        }}
        .stat.good {{ background: #c8e6c9; }}
        .stat.bad {{ background: #ffcdd2; }}
        .stat.anchor-tag {{ background: #ff9800; color: white; font-weight: bold; }}
        .sentence-text {{
            color: #333;
            line-height: 1.5;
        }}
        .final-answer {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multimodel Pipeline Report: {model_name}</h1>
        <p>Showing {len(syco_samples)} sycophantic samples and {len(non_syco_samples)} non-sycophantic samples with detected anchors.</p>
        
        {samples_html}
    </div>
    
    <script>
    function showTab(tabId, btn) {{
        const parent = btn.closest('.section');
        parent.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        btn.classList.add('active');
    }}
    </script>
</body>
</html>
'''
    
    report_path = output_dir / model_name / "report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Report generated: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for multimodel results")
    parser.add_argument("--output-dir", type=str, default="multimodel_test", help="Output directory")
    parser.add_argument("--model", type=str, default="deepseek-qwen-1.5b", help="Model name")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples per category")
    args = parser.parse_args()
    
    generate_report(Path(args.output_dir), args.model, args.n_samples)


if __name__ == "__main__":
    main()
