
    function render({ model, el }) {

        // ====================================
        // Model Catalog
        // ====================================

        const catalog = {
    ollama: (model.get("available_ollama_models") || []).map(id => ({
        label: id,
        id
    })),
    openrouter: model.get("available_openrouter_models") || []
};

        let activeProvider = model.get("selected_backend");

        const selected = new Set(
            model.get("selected_models")
                .filter(m => m.backend === activeProvider)
                .map(m => m.model)
        );

        // ====================================
        // HTML Layout
        // ====================================

        el.innerHTML = `
            <div class="judge-app">

                <header class="judge-header">
                    <div class="judge-logo">⚗️</div>
                    <div>
                        <h2>Experiment Setup</h2>
                        <p>
                            Configure your LLM-as-a-Judge experiments
                        </p>
                    </div>
                </header>

                <section class="judge-section">

                    <div class="section-title">
                        <span>🤖 Judge Models</span>
                        <span class="model-count" id="model-count">
                            0 selected
                        </span>
                    </div>

                    <p class="section-description">
                        Select a provider and one or more judge models.
                    </p>

                    <!-- Provider Selector -->

                    <div class="provider-switch">
                        <button
                            type="button"
                            class="provider-tab"
                            data-provider="ollama">
                            🖥️ Ollama
                        </button>

                        <button
                            type="button"
                            class="provider-tab"
                            data-provider="openrouter">
                            ☁️ OpenRouter
                        </button>
                    </div>

                    <!-- Model List -->

                    <div class="provider-card">
                        <h3 id="active-provider-title"></h3>
                        <div id="active-models"></div>
                    </div>


                </section>

                <section class="judge-section">
                    <div class="section-title">
                        <span>🗂️ Datasets</span>
                        <span class="model-count" id="dataset-count">0 selected</span>
                    </div>
                    <p class="section-description">Select one or more prepared dataset variants.</p>
                    <div class="dataset-list" id="dataset-list"></div>
                </section>

                <section class="judge-section">
                    <div class="section-title">
                        <span>🔬 Evaluation Methods</span>
                        <span class="model-count" id="method-count">0 selected</span>
                    </div>
                    <p class="section-description">Select standalone methods and their variants.</p>
                    <div class="methods-card">
                        <div class="method-row">
                            <span class="method-name">Baseline</span>
                            <div class="variant-buttons" id="baseline-variants"></div>
                        </div>
                        <div class="method-row method-row-second">
                            <label class="method-name second-label">
                                <input type="checkbox" id="second-enabled">
                                Second-Level
                            </label>
                        </div>
                        <div class="second-config" id="second-config">
                            <div class="second-caption">First-Level Baseline variant</div>
                            <div class="variant-buttons" id="second-variants"></div>
                            <div class="second-caption">Second-Level prompt variant</div>
                            <div class="variant-buttons" id="second-prompt-variants"></div>
                            <div class="second-hint">First-Level runs inside Second-Level automatically. Select Original and/or ADHD to compare.</div>
                        </div>
                        <div class="method-row">
                            <span class="method-name">Dynamic</span>
                            <div class="variant-buttons" id="dynamic-variants"></div>
                        </div>
                        <div class="second-config" id="dynamic-config">
                            <div class="second-caption">Based on Baseline variant</div>
                            <div class="variant-buttons" id="dynamic-baseline-variants"></div>
                            <div class="second-hint">The selected Baseline prompt is used by the Dynamic judge.</div>
                        </div>
                    </div>
                </section>

                <section class="judge-section">
                    <div class="section-title">⚙️ Experiment Options</div>
                    <div class="options-card">
                        <label class="option-row">
                            <span class="option-copy">
                                <span class="option-name">Confidence Scale</span>
                                <span class="option-description">Include confidence scoring in the judge prompt.</span>
                            </span>
                            <input type="checkbox" id="confidence-toggle" class="option-checkbox">
                        </label>
                        <label class="option-row">
                            <span class="option-copy">
                                <span class="option-name">Weights &amp; Biases</span>
                                <span class="option-description">Log experiment results and metrics.</span>
                            </span>
                            <input type="checkbox" id="wandb-toggle" class="option-checkbox">
                        </label>
                    </div>
                </section>

                <section class="judge-section">
                    <div class="section-title">📊 Experiment Overview</div>
                    <p class="section-description">Live preview of the experiment plan.</p>
                    <div class="overview-card">
                        <div class="overview-stats">
                            <div class="overview-stat"><strong id="overview-model-count">0</strong><span>Models</span></div>
                            <div class="overview-stat"><strong id="overview-dataset-count">0</strong><span>Datasets</span></div>
                            <div class="overview-stat"><strong id="overview-method-count">0</strong><span>Methods</span></div>
                            <div class="overview-stat"><strong id="overview-run-count">0</strong><span>Planned runs</span></div>
                        </div>
                        <div class="overview-detail"><span>Provider</span><strong id="overview-provider">Ollama</strong></div>
                        <div class="overview-detail"><span>Selected models</span><strong id="overview-models">None</strong></div>
                        <div class="overview-detail"><span>Selected datasets</span><strong id="overview-datasets">None</strong></div>
                        <div class="overview-detail"><span>Evaluation methods</span><strong id="overview-methods">None</strong></div>
                        <div class="overview-detail"><span>Confidence Scale</span><strong id="overview-confidence">Off</strong></div>
                        <div class="overview-detail"><span>Weights &amp; Biases</span><strong id="overview-wandb">Off</strong></div>
                        <div class="overview-note" id="overview-note">Choose a model, dataset and method to create a plan.</div>
                    </div>
                </section>
                <section class="judge-section">
                    <div class="section-title">📋 Planned Experiments</div>
                    <p class="section-description">Exact configurations that will run, in execution order. Updates automatically with your selection.</p>
                    <div class="plan-panel">
                        <div class="plan-panel-header"><strong>Execution plan</strong><span id="plan-list-count">0 runs</span></div>
                        <div class="plan-scroll"> 
                            <table class="plan-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Model</th>
                                        <th>Dataset</th>
                                        <th>Method</th>
                                        <th>Baseline</th>
                                        <th>Prediction</th>
                                        <th>Second-Level prompt</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="plan-list-body">
                                    <tr>
                                        <td colspan="8">
                                            Choose models, datasets and methods to preview runs.
                                        </td>
                                    </tr>
                                </tbody>
                            </table>

                        </div>
                    </div>
                </section>
                <section class="judge-section">
                    <div class="section-title">▶ Experiment Execution</div>
                    <p class="section-description">Review your selection before starting. Every configuration creates a separate run.</p>
                    <label class="launch-confirm"><input type="checkbox" id="launch-confirm"> I confirm execution (including possible API costs)</label>
                    <div class="execution-actions">
                        <button type="button" id="launch-experiments" class="launch-button" disabled>▶ Run Experiments</button>
                        <button type="button" id="stop-experiments" class="stop-button" disabled>■ Stop Experiments</button>
                    </div>
                    <div id="launch-status" class="launch-status" role="status" aria-live="polite">Ready</div>
                    <div id="run-progress-panel" class="run-progress-panel" hidden>
                        <div class="run-progress-header"><span id="run-progress-label">0 / 0 runs</span><span id="run-progress-percent">0%</span></div>
                        <div id="run-progress-track" class="run-progress-track" role="progressbar" aria-label="Completed experiment runs" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="run-progress-fill" class="run-progress-fill"></div></div>
                    </div>
                    <div id="current-run-label" class="current-run-label"></div>
                    <div id="sample-progress" class="current-run-label">Samples processed: —</div>
                    <div id="run-elapsed" class="current-run-label">Current run elapsed: —</div>
                    <div id="sample-activity" class="launch-hint">Last completed sample: —</div>
                    <div id="monitor-hint" class="launch-hint">Sample count updates after each processed example. The timer alone does not confirm model activity.</div>
                    <div class="launch-hint">Detailed execution logs appear below this panel.</div>
                </section>
                <section class="judge-section">
                    <div class="section-title">▦ Experiment Results</div>
                    <p class="section-description">Completed runs from this session. Select a run to inspect its metrics.</p>
                    <select id="result-picker" class="result-picker"><option value="">No completed runs yet</option></select>
                    <div id="result-detail" class="result-detail">Results will appear after the first completed run.</div>
                </section>
            </div>
        `;

        // ====================================
        // Elements
        // ====================================

        const modelContainer =
            el.querySelector("#active-models");

        const providerTitle =
            el.querySelector("#active-provider-title");

        const modelCount =
            el.querySelector("#model-count");

        // ====================================
        // Save Selection to Python
        // ====================================

        function saveSelection() {

            const values = catalog[activeProvider]
                .filter(item => selected.has(item.id))
                .map(item => ({
                    backend: activeProvider,
                    model: item.id
                }));

            model.set("selected_backend", activeProvider);
            model.set("selected_models", values);
            model.save_changes();

            modelCount.textContent =
                `${values.length} selected`;
            updateOverview();
        }

        // ====================================
        // Render Models
        // ====================================

        function renderModels() {

            modelContainer.replaceChildren();

            providerTitle.textContent =
                activeProvider === "ollama"
                    ? "Ollama Models"
                    : "OpenRouter Models";

            for (const item of catalog[activeProvider]) {

                const label =
                    document.createElement("label");

                label.className = "model-option";

                const checkbox =
                    document.createElement("input");

                checkbox.type = "checkbox";
                checkbox.checked =
                    selected.has(item.id);

                checkbox.addEventListener(
                    "change",
                    () => {

                        if (checkbox.checked) {
                            selected.add(item.id);
                        } else {
                            selected.delete(item.id);
                        }

                        saveSelection();
                    }
                );

                const text =
                    document.createElement("span");

                text.textContent = item.label;

                label.append(checkbox, text);
                modelContainer.appendChild(label);
            }

            if (activeProvider === "ollama" && catalog.ollama.length === 0) {
                modelContainer.textContent = "No installed Ollama models found (or Ollama is unavailable).";
            }

            modelCount.textContent =
                `${selected.size} selected`;
        }

        // ====================================
        // Provider Switching
        // ====================================

        function updateProviderTabs() {

            el.querySelectorAll(".provider-tab")
                .forEach(tab => {

                    tab.classList.toggle(
                        "active",
                        tab.dataset.provider === activeProvider
                    );
                });
        }

        el.querySelectorAll(".provider-tab")
            .forEach(button => {

                button.addEventListener("click", () => {

                    const nextProvider =
                        button.dataset.provider;

                    if (nextProvider === activeProvider) {
                        return;
                    }

                    activeProvider = nextProvider;

                    // One plan = one provider
                    selected.clear();


                    updateProviderTabs();
                    renderModels();
                    saveSelection();
                });
            });


        // ====================================
        // Prepared Datasets
        // ====================================

        const availableDatasets = model.get("available_datasets");
        const validIds = new Set(availableDatasets.map(d => d.dataset_id));
        const selectedDatasetIds = new Set(
            model.get("selected_datasets").map(d => d.dataset_id)
                .filter(id => validIds.has(id))
        );

        function saveDatasets() {
            const chosen = availableDatasets.filter(d => selectedDatasetIds.has(d.dataset_id));
            model.set("selected_datasets", chosen);
            model.save_changes();
            el.querySelector("#dataset-count").textContent = `${chosen.length} selected`;
            updateOverview();
        }

        const grouped = new Map();
        for (const item of availableDatasets) {
            if (!grouped.has(item.name)) grouped.set(item.name, []);
            grouped.get(item.name).push(item);
        }
        const datasetList = el.querySelector("#dataset-list");
        for (const [name, items] of grouped) {
            const row = document.createElement("div");
            row.className = "dataset-row";
            const title = document.createElement("span");
            title.className = "dataset-name";
            title.textContent = name;
            const variants = document.createElement("div");
            variants.className = "variant-buttons";
            items.sort((a,b) => a.variant.localeCompare(b.variant, undefined, {numeric:true}));
            for (const item of items) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "variant-button";
                button.textContent = item.variant;
                button.title = item.dataset_id;
                function refresh() {
                    const active = selectedDatasetIds.has(item.dataset_id);
                    button.classList.toggle("selected", active);
                    button.setAttribute("aria-pressed", String(active));
                }
                button.addEventListener("click", () => {
                    if (selectedDatasetIds.has(item.dataset_id)) selectedDatasetIds.delete(item.dataset_id);
                    else selectedDatasetIds.add(item.dataset_id);
                    refresh();
                    saveDatasets();
                });
                refresh();
                variants.appendChild(button);
            }
            row.append(title, variants);
            datasetList.appendChild(row);
        }
        if (!availableDatasets.length) {
            datasetList.textContent = "No prepared datasets found.";
        }
        el.querySelector("#dataset-count").textContent = `${selectedDatasetIds.size} selected`;

        // ====================================
        // Evaluation Methods
        // ====================================

        const promptVariants = model.get("available_prompt_variants") || {};

        const availableBaselineVariants =
            promptVariants.baseline || ["basic", "detailed"];

        const availableDynamicVariants =
            promptVariants.dynamic || ["original", "paper_based"];

        const availableSecondPromptVariants =
            promptVariants.second_level || ["original", "adhd"];

        const initialMethods = model.get("selected_methods") || [];
        const baselineVariants = new Set(
            initialMethods.filter(m => m.method === "baseline").map(m => m.baseline_criterion_detail)
        );
        const dynamicVariants = new Set(
            initialMethods.filter(m => m.method === "dynamic").map(m => m.dynamic_prediction_variant)
        );
        const dynamicBaselineVariants = new Set(
            initialMethods.filter(m => m.method === "dynamic").map(m => m.baseline_criterion_detail)
        );
        const secondVariants = new Set(
            initialMethods.filter(m => m.method === "second_level").map(m => m.baseline_criterion_detail)
        );
        const secondPromptVariants = new Set(
            initialMethods.filter(m => m.method === "second_level")
                .map(m => m.second_level_variant || "original")
        );
        let secondEnabled = secondVariants.size > 0;

        function saveMethods() {
            const values = [
                ...availableBaselineVariants.filter(v => baselineVariants.has(v))
                    .map(v => ({method:"baseline", baseline_criterion_detail:v})),
                ...(secondEnabled ? availableBaselineVariants.filter(v => secondVariants.has(v))
                    .flatMap(detail => availableSecondPromptVariants.filter(v => secondPromptVariants.has(v))
                        .map(v => ({method:"second_level", baseline_criterion_detail:detail, second_level_variant:v}))) : []),
                ...availableBaselineVariants.filter(v => dynamicBaselineVariants.has(v))
                    .flatMap(detail => availableDynamicVariants.filter(v => dynamicVariants.has(v))
                        .map(v => ({method:"dynamic", baseline_criterion_detail:detail, dynamic_prediction_variant:v})))
            ];
            model.set("selected_methods", values);
            model.save_changes();
            el.querySelector("#method-count").textContent = `${values.length} selected`;
            updateOverview();
        }

        function addVariantButtons(containerId, variants, isSelected, onClick) {
            const container = el.querySelector("#" + containerId);
            for (const value of variants) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "variant-button";
                button.textContent = value === "paper_based" ? "paper-based" : value;
                const refresh = () => {
                    const active = isSelected(value);
                    button.classList.toggle("selected", active);
                    button.setAttribute("aria-pressed", String(active));
                };
                button.addEventListener("click", () => {
                    onClick(value);
                    container.querySelectorAll("button").forEach(b => {
                        const v = b.dataset.value;
                        b.classList.toggle("selected", isSelected(v));
                        b.setAttribute("aria-pressed", String(isSelected(v)));
                    });
                    saveMethods();
                });
                button.dataset.value = value;
                refresh();
                container.appendChild(button);
            }
        }

        const toggle = (set, value) => set.has(value) ? set.delete(value) : set.add(value);
        addVariantButtons("baseline-variants", availableBaselineVariants,
            v => baselineVariants.has(v), v => toggle(baselineVariants, v));
        addVariantButtons("dynamic-variants", availableDynamicVariants,
            v => dynamicVariants.has(v), v => {
                toggle(dynamicVariants, v);
                if (dynamicVariants.size && !dynamicBaselineVariants.size) {
                    dynamicBaselineVariants.add("basic");
                    el.querySelectorAll("#dynamic-baseline-variants button").forEach(b => {
                        b.classList.toggle("selected", dynamicBaselineVariants.has(b.dataset.value));
                        b.setAttribute("aria-pressed", String(dynamicBaselineVariants.has(b.dataset.value)));
                    });
                }
            });
        addVariantButtons("dynamic-baseline-variants", availableBaselineVariants,
            v => dynamicBaselineVariants.has(v), v => toggle(dynamicBaselineVariants, v));
        addVariantButtons("second-variants", availableBaselineVariants,
            v => secondVariants.has(v), v => toggle(secondVariants, v));
        addVariantButtons("second-prompt-variants", availableSecondPromptVariants,
            v => secondPromptVariants.has(v), v => toggle(secondPromptVariants, v));

        const secondCheck = el.querySelector("#second-enabled");
        const secondConfig = el.querySelector("#second-config");
        secondCheck.checked = secondEnabled;
        function updateSecond() {
            secondConfig.hidden = !secondEnabled;
            if (secondEnabled && secondVariants.size === 0) {
                secondVariants.add(availableBaselineVariants[0] || "basic");
            }
            if (secondEnabled && secondPromptVariants.size === 0) {
                secondPromptVariants.add("original");
            }
            for (const [id, values] of [["second-variants", secondVariants],
                                         ["second-prompt-variants", secondPromptVariants]]) {
                el.querySelectorAll(`#${id} button`).forEach(b => {
                    const active = values.has(b.dataset.value);
                    b.classList.toggle("selected", active);
                    b.setAttribute("aria-pressed", String(active));
                });
            }
        }
        secondCheck.addEventListener("change", () => {
            secondEnabled = secondCheck.checked;
            updateSecond();
            saveMethods();
        });
        updateSecond();
        el.querySelector("#method-count").textContent =
            `${baselineVariants.size + dynamicBaselineVariants.size * dynamicVariants.size + (secondEnabled ? secondVariants.size * secondPromptVariants.size : 0)} selected`;

        // ====================================
        // Experiment Options
        // ====================================

        const confidenceToggle = el.querySelector("#confidence-toggle");
        const wandbToggle = el.querySelector("#wandb-toggle");
        confidenceToggle.checked = model.get("confidence_mode");
        wandbToggle.checked = model.get("use_wandb");

        confidenceToggle.addEventListener("change", () => {
            model.set("confidence_mode", confidenceToggle.checked);
            model.save_changes();
            updateOverview();
        });
        wandbToggle.addEventListener("change", () => {
            model.set("use_wandb", wandbToggle.checked);
            model.save_changes();
            updateOverview();
        });

        // ====================================
        // Live Experiment Overview
        // ====================================
        
        function isMethodAvailable(dataset, method) {
            const criterion = dataset.evaluation_criterion;

            const baselineVariants =
                (promptVariants.by_criterion || {})[criterion] || [];

            if (!baselineVariants.includes(method.baseline_criterion_detail)) {
                return false;
            }

            if (method.method === "dynamic") {
                const dynamicVariants =
                    (promptVariants.dynamic_by_criterion || {})[criterion] || [];

                return dynamicVariants.includes(
                    method.dynamic_prediction_variant
                );
            }

            return true;
        }

        function updateOverview() {
            const models = catalog[activeProvider].filter(item => selected.has(item.id));
            const datasets = availableDatasets.filter(item => selectedDatasetIds.has(item.dataset_id));
            const methods = [
                ...availableBaselineVariants.filter(v => baselineVariants.has(v))
                    .map(v => `Baseline (${v})`),
                ...(secondEnabled ? availableBaselineVariants.filter(v => secondVariants.has(v))
                    .flatMap(detail => availableSecondPromptVariants.filter(v => secondPromptVariants.has(v))
                        .map(v => `Second-Level (${detail} first-level, ${v} prompt)`)) : []),
                ...availableBaselineVariants.filter(v => dynamicBaselineVariants.has(v))
                    .flatMap(detail => availableDynamicVariants.filter(v => dynamicVariants.has(v))
                        .map(v => `Dynamic (${detail} baseline, ${v.replace("_", "-")})`))
            ];
            
            const selectedMethodConfigs = model.get("selected_methods") || [];

            const validCombinations = datasets.flatMap(dataset =>
                selectedMethodConfigs
                    .filter(method => isMethodAvailable(dataset, method))
                    .map(method => ({ dataset, method }))
            );

            const count = models.length * validCombinations.length;

            const skippedCount =
                models.length *
                (datasets.length * selectedMethodConfigs.length -
                validCombinations.length);

            const put = (id, value) => { el.querySelector("#" + id).textContent = String(value); };

            put("overview-model-count", models.length);
            put("overview-dataset-count", datasets.length);
            put("overview-method-count", methods.length);
            put("overview-run-count", count);
            put("overview-provider", activeProvider === "ollama" ? "Ollama" : "OpenRouter");
            put("overview-models", models.map(m => m.label).join(", ") || "None");
            put("overview-datasets", datasets.map(d => d.dataset_id).join(", ") || "None");
            put("overview-methods", methods.join(", ") || "None");
            put("overview-confidence", confidenceToggle.checked ? "On" : "Off");
            put("overview-wandb", wandbToggle.checked ? "On" : "Off");

            // The order matches Python build_plan(): models × datasets × methods.
            // Use textContent rather than HTML interpolation for user-provided names.
            const planBody = el.querySelector("#plan-list-body");
            const planCount = el.querySelector("#plan-list-count");
            planBody.replaceChildren();
            const chosenMethods = model.get("selected_methods") || [];
            const chosenModels = model.get("selected_models") || [];
            const chosenDatasets = model.get("selected_datasets") || [];
            
            const totalRuns = chosenModels.length *
                chosenDatasets.reduce((total, dataset) =>
                    total + chosenMethods.filter(
                        method => isMethodAvailable(dataset, method)
                    ).length,
                0);

            planCount.textContent = `${totalRuns} run${totalRuns === 1 ? "" : "s"}`;
            if (chosenModels.length === 0 ||
                chosenDatasets.length === 0 ||
                chosenMethods.length === 0) {
                const row = document.createElement("tr");
                const cell = document.createElement("td");
                cell.colSpan = 8;
                cell.textContent = "Choose models, datasets and methods to preview runs.";
                row.appendChild(cell);
                planBody.appendChild(row);
            } else {
                let number = 0;
                for (const chosenModel of chosenModels) {
                    for (const dataset of chosenDatasets) {
                        for (const method of chosenMethods) {
                            number += 1;
                            const row = document.createElement("tr");
                            const label = catalog[activeProvider].find(m => m.id === chosenModel.model)?.label || chosenModel.model;
                            const methodName = method.method === "second_level" ? "Second-Level" :
                                method.method === "dynamic" ? "Dynamic" : "Baseline";
                            
                            const isAvailable = isMethodAvailable(dataset, method);

                            if (!isAvailable) {
                                row.style.opacity = "0.5";
                                row.title = `Variant '${method.baseline_criterion_detail}' is not available for this dataset.`;
                            }


                            const cells = [
                                number,
                                label,
                                dataset.dataset_id,
                                methodName,
                                method.baseline_criterion_detail || "—",
                                method.method === "dynamic"
                                    ? (method.dynamic_prediction_variant || "—")
                                    : "—",
                                method.method === "second_level"
                                    ? (method.second_level_variant || "original")
                                    : "—",
                                isAvailable ? "Ready" : "Unavailable"
                            ];

                            for (const value of cells) {
                                const cell = document.createElement("td");
                                cell.textContent = String(value);
                                row.appendChild(cell);
                            }
                            planBody.appendChild(row);
                        }
                    }
                }
            }

            const note = el.querySelector("#overview-note");
            
            note.textContent = count
                ? `${count} independent run${count === 1 ? "" : "s"} planned.` +
                (skippedCount > 0
                    ? ` ${skippedCount} unsupported combination(s) skipped.`
                    : " Second-Level includes its own First-Level calls.")
                : skippedCount > 0
                    ? `No valid runs. ${skippedCount} unsupported combination(s) skipped.`
                    : "Choose at least one model, dataset and method to create a plan.";

        }

        // ====================================
        // Initial Render
        // ====================================

        updateProviderTabs();
        renderModels();
        updateOverview();
        // Unified launch controls inside Experiment Setup
        const launchButton = el.querySelector("#launch-experiments");
        const stopButton = el.querySelector("#stop-experiments");
        const launchConfirm = el.querySelector("#launch-confirm");
        const launchStatus = el.querySelector("#launch-status");
        const progressPanel = el.querySelector("#run-progress-panel");
        const progressLabel = el.querySelector("#run-progress-label");
        const progressPercent = el.querySelector("#run-progress-percent");
        const progressFill = el.querySelector("#run-progress-fill");
        const progressTrack = el.querySelector("#run-progress-track");
        const updateProgress = () => {
            const total = model.get("run_total") || 0;
            const current = model.get("run_current") || 0;
            const active = model.get("run_active");
            const percent = total ? Math.round(100 * current / total) : 0;
            progressPanel.hidden = total === 0;
            progressLabel.textContent = `${current} / ${total} runs completed${active ? " · Running" : ""}`;
            progressPercent.textContent = `${percent}%`;
            progressFill.style.width = `${percent}%`;
            progressTrack.setAttribute("aria-valuenow", String(percent));
        };
        for (const name of ["run_total", "run_current", "run_active"]) {
            model.on(`change:${name}`, updateProgress);
        }
        updateProgress();
        const refreshLaunch = () => {
            
            const chosenModels = model.get("selected_models") || [];
            const chosenDatasets = model.get("selected_datasets") || [];
            const chosenMethods = model.get("selected_methods") || [];

            const planned = chosenModels.length *
                chosenDatasets.reduce((total, dataset) =>
                    total + chosenMethods.filter(
                        method => isMethodAvailable(dataset, method)
                    ).length,
                0);

            launchButton.disabled = !launchConfirm.checked || planned === 0 || model.get("run_active") || model.get("execution_busy");
            stopButton.disabled = !model.get("run_active");
            launchButton.textContent = planned ? `▶ Run Experiments (${planned})` : "▶ Run Experiments";
        };
        launchConfirm.addEventListener("change", refreshLaunch);
        for (const name of ["selected_models", "selected_datasets", "selected_methods"]) {
            model.on(`change:${name}`, () => { launchConfirm.checked = false; refreshLaunch(); });
        }
        for (const name of ["confidence_mode", "use_wandb", "selected_backend"]) {
            model.on(`change:${name}`, () => { launchConfirm.checked = false; refreshLaunch(); });
        }
        model.on("change:run_status", () => {
            launchStatus.textContent = model.get("run_status");
            refreshLaunch();
        });
        launchButton.addEventListener("click", () => {
            if (launchButton.disabled || !launchConfirm.checked) return;
            launchConfirm.checked = false;
            launchButton.disabled = true;
            model.set("launch_request", model.get("launch_request") + 1);
            model.save_changes();
        });
        stopButton.addEventListener("click", () => {
            if (stopButton.disabled) return;
            stopButton.disabled = true;
            launchStatus.textContent = "Stop requested — waiting for current run…";
            model.set("stop_request", model.get("stop_request") + 1);
            model.save_changes();
        });

        const sampleLabel = el.querySelector("#sample-progress");
        const activityLabel = el.querySelector("#sample-activity");
        const renderSamples = () => {
            const done = model.get("sample_current") || 0;
            const total = model.get("sample_total") || 0;
            sampleLabel.textContent = total
                ? `Samples processed: ${done}/${total} (${(100*done/total).toFixed(1)}%)`
                : "Samples processed: —";
            const last = model.get("sample_last_completed_at") || 0;
            activityLabel.textContent = last
                ? `Last completed sample: ${new Date(last*1000).toLocaleTimeString()}`
                : "Last completed sample: —";
        };
        ["sample_current","sample_total","sample_last_completed_at"].forEach(
            key => model.on(`change:${key}`, renderSamples)
        );
        renderSamples();
        const elapsedLabel = el.querySelector("#run-elapsed");
        const tickElapsed = () => {
            if (!el.isConnected) return;
            const start = model.get("current_run_started_at") || 0;
            const active = model.get("run_active");
            if (!start || !active) {
                elapsedLabel.textContent = "Current run elapsed: —";
                return;
            }
            const elapsed = Math.max(0, Math.floor(Date.now()/1000 - start));
            const mm = String(Math.floor(elapsed/60)).padStart(2,"0");
            const ss = String(elapsed%60).padStart(2,"0");
            elapsedLabel.textContent = `Current run elapsed: ${mm}:${ss} · Awaiting sample/run completion`;
        };
        const timerId = setInterval(() => {
            if (!el.isConnected) { clearInterval(timerId); return; }
            tickElapsed();
        }, 1000);
        model.on("change:current_run_started_at", tickElapsed);
        tickElapsed();
        const currentRunLabel = el.querySelector("#current-run-label");
        const resultPicker = el.querySelector("#result-picker");
        const resultDetail = el.querySelector("#result-detail");
        const fmt = (value, percent=false) => {
            if (value === null || value === undefined || !Number.isFinite(Number(value))) return "N/A";
            return percent ? (100 * Number(value)).toFixed(1) + "%" : Number(value).toFixed(3);
        };
        function renderResults() {
            const results = model.get("run_results") || [];
            const previous = resultPicker.value;
            resultPicker.replaceChildren();
            if (!results.length) {
                const option = document.createElement("option");
                option.value = "";
                option.textContent = "No completed runs yet";
                resultPicker.append(option);
                resultDetail.textContent = "Results will appear after the first completed run.";
                return;
            }
            results.forEach((r, index) => {
                const option = document.createElement("option");
                option.value = String(index);
                option.textContent = `${index+1}. ${r.model} · ${r.dataset_id} · ${r.method} (${r.baseline_detail}${r.method === "second_level" ? ", " + (r.second_level_variant || "original") : ""})`;
                resultPicker.append(option);
            });
            resultPicker.value = previous && Number(previous) < results.length ? previous : String(results.length-1);
            showResult();
        }
        function showResult() {
            const results = model.get("run_results") || [];
            const r = results[Number(resultPicker.value)];
            resultDetail.replaceChildren();
            if (!r) return;
            const headline = document.createElement("div");
            headline.className = "result-headline";
            headline.textContent = `${r.model} · ${r.dataset_id} · ${r.method} (${r.baseline_detail}${r.method === "second_level" ? ", " + (r.second_level_variant || "original") : ""})`;
            resultDetail.append(headline);
            const metrics = document.createElement("div");
            metrics.className = "result-metrics";
            [["Accuracy",fmt(r.accuracy,true)],["Macro F1",fmt(r.macro_f1)],["Coverage",fmt(r.coverage,true)],
             ["Valid outputs",`${r.valid_samples}/${r.total_samples}`],["Runtime errors",String(r.runtime_errors)]].forEach(([k,v]) => {
                const card = document.createElement("div"); card.className="result-metric";
                const label=document.createElement("small"); label.textContent=k;
                const val=document.createElement("strong"); val.textContent=v;
                card.append(label,val); metrics.append(card);
            });
            resultDetail.append(metrics);
            const cmTitle = document.createElement("div");
            cmTitle.className = "result-subtitle"; cmTitle.textContent = "Confusion Matrix";
            resultDetail.append(cmTitle);
            if (r.confusion_matrix && r.confusion_matrix.length === 2) {
                const table = document.createElement("table");
                table.className = "result-matrix";
                const rows = [
                    ["","Pred. negative","Pred. positive"],
                    ["Actual negative",r.confusion_matrix[0][0],r.confusion_matrix[0][1]],
                    ["Actual positive",r.confusion_matrix[1][0],r.confusion_matrix[1][1]],
                ];
                rows.forEach((cells,i) => {
                    const tr=document.createElement("tr");
                    cells.forEach((value,j) => {
                        const td=document.createElement(i===0||j===0?"th":"td");
                        td.textContent=String(value); tr.append(td);
                    });
                    table.append(tr);
                });
                resultDetail.append(table);
            } else {
                const none=document.createElement("p");
                none.textContent="N/A — no valid predictions or no binary confusion matrix.";
                resultDetail.append(none);
            }
            const wandb=document.createElement("p");
            wandb.className="result-link";
            if (r.wandb_url && /^https:\/\/wandb\.ai\//.test(r.wandb_url)) {
                const a=document.createElement("a");
                a.href=r.wandb_url; a.target="_blank"; a.rel="noopener noreferrer";
                a.textContent="Open this run in Weights & Biases ↗";
                wandb.append(a);
            } else {
                wandb.textContent=`W&B: ${r.wandb_status || "not available"}`;
            }
            resultDetail.append(wandb);
        }
        resultPicker.addEventListener("change",showResult);
        model.on("change:run_results",renderResults);
        model.on("change:current_run_label",() => {
            currentRunLabel.textContent=model.get("current_run_label");
        });
        renderResults();
        model.on("change:run_active", refreshLaunch);
        model.on("change:execution_busy", refreshLaunch);
        refreshLaunch();

    }

    export default { render };