# Dataset dictionary

The file `data/shipments_synthetic.csv` contains 300 simulated shipment journeys.

| Field | Meaning |
|---|---|
| shipment_id | Unique synthetic shipment identifier |
| origin / destination | Simulated route endpoints |
| route_region | Broad route grouping |
| scheduled_* | Planned stage timestamps |
| actual_* | Simulated realized stage timestamps |
| transit_days | Planned origin-to-port transit duration |
| origin_departure_delay_hours | Departure delay observed at origin |
| port_processing_delay_hours | Port processing delay |
| customs_delay_hours | Customs clearance delay |
| port_congestion_index | Synthetic congestion score from 0 to 1 |
| weather_severity_index | Synthetic weather severity score from 0 to 1 |
| peak_season | Indicator for selected peak shipping months |
| cumulative_observed_delay_hours | Sum of available stage delay observations; missing readings are skipped |
| port_delay_congestion_interaction | Port delay multiplied by congestion |
| weather_congestion_interaction | Weather severity multiplied by congestion |
| final_delay_hours | Target: actual final delivery minus scheduled final delivery, in hours |
| delayed_over_24h | Target: 1 if final delay exceeds 24 hours, otherwise 0 |

**Prediction-time contract:** Features are intended to be available after customs clearance and before final delivery. `actual_delivery`, `final_delay_hours`, and `delayed_over_24h` are outcomes and are never used as model inputs.
