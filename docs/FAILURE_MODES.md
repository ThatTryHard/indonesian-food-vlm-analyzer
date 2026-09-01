# Robustness and Failure Modes

| Failure | Detection | Required behavior |
|---|---|---|
| Corrupt/unreadable image | PIL verification/load error | Log and exclude before sampling |
| Duplicate/near-duplicate | SHA-256 and perceptual hash | Keep one representative; one duplicate group cannot cross splits |
| Non-food image | Human flag or future detector | Abstain |
| Unknown/fusion dish | Low evidence or VLM abstention | Do not force a known class |
| Multiple dishes | Annotation note and component labels | Treat image as multi-component; avoid dish-level nutrition |
| Occlusion/poor lighting | Human uncertainty/quality flag | Use uncertain labels or abstain |
| Hidden ingredient | No direct visual evidence | Exclude from visible ground truth |
| VLM malformed JSON | Strict parser status | Count parse failure; do not silently repair into a prediction |
| VLM unknown label | Ontology validation | Count schema violation; do not add an alias after test |
| Impossible nutrition | Macro/Atwater validator | Reject output; no display |
| Data or concept drift | Class/label/abstention distribution | Re-annotate and recalibrate before continued use |

The portfolio target does not include a deployed service, so latency and availability objectives remain intentionally unspecified rather than guessed.
