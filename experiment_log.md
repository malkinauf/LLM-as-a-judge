### This document summarizes conducted experiments, highlighting key findings and guiding further improvements.
## exp_01_baseline_second_level
1. ~50% parsing errors
2. Second level always "correct"
3. 9 disagreements (true vs first-level)
<br> ### → issue: unreliable JSON output
<br> ### → next: improve prompt format
---
## exp_02_dynamic_prompt
93% parsing errors: missing closing JSON brace  
<br> ### → next: fix prompt format or check parsing code
