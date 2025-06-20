from flask import Flask, request, jsonify
import os
import json
from datetime import datetime
from main_agentic_flow import AutoBMCSystem, WorkflowState, BMCVersion

app = Flask(__name__)

def reconstruct_state(data):
    # Reconstruct WorkflowState from dict (simplified, assumes correct format)
    state = WorkflowState(**{k: v for k, v in data.items() if k in WorkflowState.__fields__})
    # Rebuild version_history as BMCVersion objects
    if 'version_history' in data:
        state.version_history = [BMCVersion(**v) for v in data['version_history']]
    return state

@app.route('/rollback-version', methods=['POST'])
def rollback_version():
    try:
        req = request.get_json()
        version_id = req.get('version_id')
        state_data = req.get('state')
        if not version_id or not state_data:
            return jsonify({'error': 'version_id and state are required'}), 400

        # Reconstruct state
        state = reconstruct_state(state_data)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        system = AutoBMCSystem(gemini_api_key=gemini_api_key)

        # Rollback
        state = system.rollback_to_version(state, version_id)
        if state.current_step == "rollback_completed":
            # Create a new version entry for the rollback
            version_id_new = f"v{len(state.version_history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            timestamp = datetime.now().isoformat()
            new_version = BMCVersion(
                version_id=version_id_new,
                timestamp=timestamp,
                bmc_data=state.bmc_draft,
                validation_report=state.validation_report,
                changes_made=[f"Rollback to {version_id}"]
            )
            state.version_history.append(new_version)
            if len(state.version_history) > 10:
                state.version_history = state.version_history[-10:]
            state.current_step = "rollback_version_created"
            response = {
                'success': True,
                'bmc_draft': state.bmc_draft.dict() if state.bmc_draft else None,
                'validation_report': state.validation_report.dict() if state.validation_report else None,
                'version_history': [v.dict() for v in state.version_history],
                'current_step': state.current_step,
                'errors': state.errors
            }
        else:
            response = {
                'success': False,
                'errors': state.errors
            }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 