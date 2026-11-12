# Finish the 7.3.4 Shiny training workflow and checkpoint feedback

The workbench now guides users through Data, Features, Build & Train, and Model / Results, with Experiments retained separately. Training and downloads follow the state of the current browser session: Train requires prepared data and no active job, Cancel requires a job, Save requires a completed model, and new-prediction downloads require scored rows. Existing completed models remain usable while another model trains. Warm starts and model/data replacement wait for the current job to finish or be cancelled.

New fits default to the best validation checkpoint (classification accuracy or regression MSE), restoring both weights and biases before inference; final-epoch weights remain an explicit alternative. Both choices run all requested epochs. Summaries identify the selected epoch, fit budget, selection mode, criterion and validation score. Initial self-organization runs only for new models. Equivalent R panels and downloaded recipes retain the completed model's choices and append warm starts without repeating initial self-organization.

Training failures offer actionable guidance and expandable worker diagnostics. Failed and cancelled jobs retain the previous completed model. Loss-chart hover reports epoch and loss without selecting weights; an expandable table supports keyboards and touch screens. Small lowercase-i help controls and right-aligned switches remain keyboard accessible.

Validation and build/check results are recorded in the accompanying validation report. This remains version 7.3.4. No optimizer implementations, automatic early stopping, release tags, merges or publication are part of this change.
