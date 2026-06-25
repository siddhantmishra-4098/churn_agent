from typing import TypedDict, Optional

class AgentState(TypedDict):
    query:              str
    csv_path:           Optional[str]
    customer_id:        Optional[int]
    route:              Optional[str]
    rf_churn_prob:      Optional[float]
    rf_label:           Optional[str]
    aft_churn_prob:     Optional[float]
    prediction_result:  Optional[str]
    customer_info:      Optional[str]
    explanation:        Optional[str]
    error:              Optional[str]