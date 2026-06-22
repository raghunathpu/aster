import urllib.parse
import sys
import textwrap

# Set encoding to utf-8 for stdout if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

with open('app/aster_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract lines 1296 to 1304 (0-indexed: 1295 to 1303)
block = "".join(lines[1295:1304])
block = textwrap.dedent(block)
print("Extracting block (dedented):")
print(block)

corridor = "Mysore Road"
nearest_junc = "toll gate mysore road"
cause_key = "vehicle_breakdown"
res_time = 45
event_hour = 8
time_hr = "08:00 IST"
closure_line = ""
eff_tier = "Medium"
route_text = "Divert via alternative routes"
CAUSE_LABELS = { "vehicle_breakdown": "Vehicle Breakdown" }

local_vars = locals()
exec(block, globals(), local_vars)
wa_text_val = local_vars['wa_text']

print("\nEvaluated wa_text:")
print(repr(wa_text_val))

quoted = urllib.parse.quote(wa_text_val)
print("\nQuoted wa_text:")
print(quoted)
