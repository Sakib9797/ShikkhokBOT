import json

dataset_dir = 'data/SSC-BanglaTutor'

subjects = [
    ('Bio/SSC_Biology_Datasets.jsonl', 'Biology'),
    ('chem/SSC_Chemistry_Dataset.jsonl', 'Chemistry'),
    ('phy/SSC_Physics_Dataset.jsonl', 'Physics'),
]

output = []
for path, name in subjects:
    with open(f'{dataset_dir}/{path}', encoding='utf-8') as f:
        lines = f.readlines()

    output.append(f"{'='*60}")
    output.append(f"  {name}: {len(lines)} entries")
    output.append(f"{'='*60}")

    # Show structure from first entry
    entry = json.loads(lines[0])
    output.append(f"\nKeys: {list(entry.keys())}\n")
    output.append(f"Question: {entry['Question'][:100]}...\n")
    output.append(f"Hints ({len(entry['Hints'])} levels):")
    for i, h in enumerate(entry['Hints']):
        output.append(f"  Hint {i+1}: {h[:80]}...")
    output.append(f"\nExactAnswer: {entry['ExactAnswer']}")
    output.append(f"\nCandidates_Answers: {entry['Candidates_Answers']}")
    output.append(f"\nConvergence: {json.dumps(entry['Convergence'][:3], indent=2, ensure_ascii=False)}")
    output.append(f"\nConvergence_Ranked: {str(entry.get('Convergence_Ranked', 'N/A'))[:200]}")
    output.append(f"\nTopicTags: {entry.get('TopicTags', 'N/A')}")
    output.append("")

with open('ssc_preview.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print('Preview written to ssc_preview.txt')
