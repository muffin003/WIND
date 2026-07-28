(() => {
  const localRun = `cd "path\\to\\WIND"
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gym,dev]"
python -m wind_benchmark.web`;

  window.WIND_GUIDES = {
    en: {
      labels: { guide: "REFERENCE", contents: "Page guide", copy: "COPY", source: "SOURCE", use: "When to use", caution: "Keep in mind", parameters: "Coefficient guide", scenarios: "Interactive examples", collapse: "Collapse guide", expand: "Expand guide" },
      pages: {
        overview: {
          title: "How WIND works",
          intro: "A compact reference for the benchmark protocol, package architecture and local setup.",
          topics: [
            { title: "Benchmark objective", body: "WIND compares how optimization algorithms track a hidden moving optimum under changing landscapes, drift, noise and information constraints.", formula: String.raw`x_t\longrightarrow\mathcal O_t(x_t)\longrightarrow\theta_{t+1}` },
            { title: "Interaction protocol", body: "At step t the optimizer commits to xₜ, receives only the configured oracle observation, and then the environment advances. Ground-truth θₜ is recorded for evaluation but is never exposed to the optimizer.", bullets: ["No look-ahead into the next environment state", "First- and zero-order information barriers", "Identical seeds reproduce drift and noise"] },
            { title: "Package architecture", body: "core.py defines environments; oracle.py controls feedback; benchmark.py runs evaluations; metrics.py computes scores; experiment.py contains reference optimizers; gym_env.py exposes the RL adapter." },
            { title: "Scientific guarantees", body: "The test suite checks landscape invariants, temporal consistency, information barriers, metric definitions, result export and deterministic reset behavior.", bullets: ["θₜ is not exposed in Observation", "Regret uses the matching environment state", "Result metadata is sufficient for verification"] },
            { title: "Run locally after downloading", body: "Open PowerShell in the downloaded repository and execute these commands. The last command starts the page at http://127.0.0.1:8765.", code: localRun },
          ],
          scenarios: true,
        },
        environment: {
          title: "Environment reference",
          intro: "Choose the geometry of the objective and the law that moves its hidden optimum.",
          topics: [
            { title: "Landscape", body: "A landscape defines fₜ(x), the feasible geometry and the meaning of distance. Selecting a card reveals its formula and every editable coefficient." },
            { title: "Drift", body: "A drift updates θₜ only after the optimizer has acted. Stationary, smooth, stochastic, abrupt, adaptive and manifold-preserving dynamics test different tracking abilities." },
            { title: "Initial state and bounds", body: "A scalar initial value is broadcast to all coordinates. A vector must contain exactly dim entries. Euclidean tasks use box bounds; manifold tasks enforce their own geometry." },
          ],
        },
        oracle: {
          title: "Observation reference",
          intro: "The oracle is the only information channel between the optimizer and the hidden environment.",
          topics: [
            { title: "Feedback modes", body: "First-order exposes a gradient, zero-order exposes values, hybrid provides both, scheduled alternates modes, and offline replay uses a recorded θ trajectory." },
            { title: "Information barrier", body: "Blind-value first-order mode removes fₜ(xₜ) from the observation. Optimizers never receive θₜ, clean noise values or future states." },
            { title: "Noise channels", body: "Value and gradient noise are configured independently. Their coefficients belong to the observation model and do not change the clean landscape used for ground-truth metrics." },
          ],
        },
        optimizers: {
          title: "Optimizer reference",
          intro: "The catalog contains reference implementations, not a ranking. Select any number of compatible methods to compare them with the same environment and seeds.",
          topics: [
            { title: "First-order methods", body: "These methods require gradients. They cover plain and momentum SGD, adaptive moments, mirror descent, dual averaging, proximal updates and sign compression." },
            { title: "Zero-order methods", body: "These methods estimate useful directions from function values. Their formulas explicitly separate perturbation, measurement, estimation and update stages." },
            { title: "Fair comparison", body: "Use identical seeds, horizon, environment and oracle policy. Query efficiency should be inspected alongside tracking quality because finite-difference methods can require multiple observations." },
            { title: "Custom methods", body: "A custom method implements OptimizerProtocol, provides reset() and step(observation), and declares whether it requires first- or zero-order feedback. No base-class inheritance is required." },
          ],
        },
        runner: {
          title: "Execution and metrics",
          intro: "Configure reproducible runs, choose evaluation metrics and export a complete experiment description.",
          topics: [
            { title: "Seeds and repetitions", body: "Every optimizer is run for every selected seed. Multiple seeds measure variability instead of reporting a single favorable trajectory.", formula: String.raw`N_{\mathrm{runs}}=N_{\mathrm{optimizers}}N_{\mathrm{seeds}}` },
            { title: "Metrics", body: "Tracking error measures distance to θₜ; dynamic regret measures excess loss; recovery and adaptation metrics describe response to changes; query efficiency accounts for oracle cost." },
            { title: "Reproducible record", body: "Keep the configuration JSON, result files, Git commit and Python dependency versions. The fixed inspector is the exact configuration currently being edited." },
            { title: "CLI and Python API", body: "The same components can be constructed from Python dictionaries or executed from a saved experiment JSON. Use dry-run mode to inspect a resolved grid before calculations start." },
          ],
        },
        results: {
          title: "Result files",
          intro: "Local engine runs produce JSON summaries and optional long-format CSV trajectories. Static mode can inspect files selected from your computer.",
          topics: [
            { title: "JSON", body: "JSON stores optimizer and environment metadata, seeds, status, final metrics and—when enabled—the full trajectory required for verification." },
            { title: "CSV and local analysis", body: "CSV is useful for external plotting and statistical tools. Loading either format happens in the browser; the file is not uploaded to GitHub Pages." },
          ],
        },
        analysis: {
          title: "Reading an experiment",
          intro: "Analysis becomes available only after a JSON or CSV result is loaded.",
          topics: [
            { title: "Time and recovery", body: "Inspect the entire curve, transient response after drift changes and the tail window. A final scalar alone can hide instability." },
            { title: "Across seeds and algorithms", body: "Compare distributions and uncertainty across identical seeds. Trajectory views should be interpreted using the selected Euclidean or manifold geometry." },
          ],
        },
        gym: {
          title: "Gymnasium adapter",
          intro: "WindGymEnv presents the same dynamic optimization task as an RL environment without weakening the oracle information barrier.",
          topics: [
            { title: "Actions", body: "Absolute actions propose the next point directly. Delta actions propose a bounded displacement. Geometry projection keeps actions feasible." },
            { title: "Rewards", body: "Negative regret measures excess objective value; negative error measures geometry-aware distance to the hidden optimum for evaluation." },
            { title: "Constrained geometry", body: "Simplex actions are projected to probabilities. Stiefel uses orthonormal frames; Grassmann treats bases related by an orthogonal transform as the same subspace." },
          ],
        },
      },
    },
    ru: {
      labels: { guide: "СПРАВКА", contents: "Материалы страницы", copy: "КОПИРОВАТЬ", source: "КОД", use: "Когда использовать", caution: "Обратите внимание", parameters: "Коэффициенты", scenarios: "Интерактивные примеры", collapse: "Свернуть справку", expand: "Развернуть справку" },
      pages: {
        overview: {
          title: "Как устроен WIND",
          intro: "Краткое пособие по протоколу бенчмарка, архитектуре пакета и локальному запуску.",
          topics: [
            { title: "Задача бенчмарка", body: "WIND сравнивает, как алгоритмы отслеживают скрытый движущийся оптимум при разных ландшафтах, дрейфах, шумах и информационных ограничениях.", formula: String.raw`x_t\longrightarrow\mathcal O_t(x_t)\longrightarrow\theta_{t+1}` },
            { title: "Протокол взаимодействия", body: "На шаге t оптимизатор фиксирует xₜ, получает только разрешённое наблюдение оракула, после чего среда переходит к следующему состоянию. Истинное θₜ сохраняется для оценки, но не передаётся алгоритму.", bullets: ["Нет доступа к будущему состоянию среды", "Разделение first-order и zero-order информации", "Одинаковые seeds воспроизводят дрейф и шум"] },
            { title: "Архитектура пакета", body: "core.py задаёт среды; oracle.py управляет обратной связью; benchmark.py выполняет запуски; metrics.py считает показатели; experiment.py содержит примеры оптимизаторов; gym_env.py реализует RL-адаптер." },
            { title: "Научные гарантии", body: "Тесты проверяют инварианты ландшафтов, временную согласованность, информационный барьер, определения метрик, экспорт и детерминированный reset.", bullets: ["θₜ не передаётся внутри Observation", "Regret использует соответствующее состояние среды", "Метаданных результата достаточно для проверки"] },
            { title: "Локальный запуск после скачивания", body: "Откройте PowerShell в скачанном репозитории и выполните команды ниже. Последняя команда откроет страницу на http://127.0.0.1:8765.", code: localRun },
          ],
          scenarios: true,
        },
        environment: {
          title: "Справка по среде",
          intro: "Выберите геометрию целевой функции и закон движения её скрытого оптимума.",
          topics: [
            { title: "Ландшафт", body: "Ландшафт задаёт fₜ(x), допустимую геометрию и смысл расстояния. При выборе карточки раскрываются формула и все изменяемые коэффициенты." },
            { title: "Дрейф", body: "Дрейф обновляет θₜ только после действия оптимизатора. Стационарная, плавная, случайная, скачкообразная, адаптивная и многообразная динамики проверяют разные свойства отслеживания." },
            { title: "Начальное состояние и границы", body: "Одно число начального состояния применяется ко всем координатам. Вектор должен содержать ровно dim значений. Евклидовы задачи используют box-границы, а многообразия — собственные ограничения." },
          ],
        },
        oracle: {
          title: "Справка по наблюдениям",
          intro: "Оракул — единственный информационный канал между оптимизатором и скрытой средой.",
          topics: [
            { title: "Режимы обратной связи", body: "First-order передаёт градиент, zero-order — значения, hybrid — оба типа данных, scheduled чередует режимы, а offline replay воспроизводит сохранённую траекторию θ." },
            { title: "Информационный барьер", body: "Blind-value режим скрывает fₜ(xₜ). Оптимизатор никогда не получает θₜ, чистую реализацию шума или будущие состояния." },
            { title: "Каналы шума", body: "Шум значения и градиента задаётся независимо. Он изменяет наблюдение, но не чистый ландшафт, по которому вычисляются контрольные метрики." },
          ],
        },
        optimizers: {
          title: "Справка по оптимизаторам",
          intro: "Каталог содержит примеры реализаций, а не рейтинг. Можно выбрать несколько совместимых методов и сравнить их на одинаковых средах и seeds.",
          topics: [
            { title: "First-order методы", body: "Этим методам нужен градиент: SGD и импульсные варианты, адаптивные моменты, зеркальный спуск, dual averaging, proximal-обновления и sign compression." },
            { title: "Zero-order методы", body: "Эти методы строят направление только по значениям функции. Формулы разделены на генерацию возмущения, измерения, оценку и обновление." },
            { title: "Честное сравнение", body: "Используйте одинаковые seeds, горизонт, среду и политику оракула. Качество отслеживания следует рассматривать вместе с числом запросов." },
            { title: "Собственные методы", body: "Пользовательский метод реализует OptimizerProtocol, методы reset() и step(observation), а также объявляет необходимый тип обратной связи. Наследование от базового класса не требуется." },
          ],
        },
        runner: {
          title: "Запуск и метрики",
          intro: "Настройте воспроизводимые запуски, выберите показатели и сохраните полное описание эксперимента.",
          topics: [
            { title: "Seeds и повторения", body: "Каждый оптимизатор запускается для каждого seed. Несколько seeds показывают вариативность, а не одну удачную траекторию.", formula: String.raw`N_{\mathrm{runs}}=N_{\mathrm{optimizers}}N_{\mathrm{seeds}}` },
            { title: "Метрики", body: "Tracking error измеряет расстояние до θₜ; dynamic regret — избыточный loss; recovery и adaptation описывают реакцию на изменения; query efficiency учитывает стоимость запросов." },
            { title: "Воспроизводимая запись", body: "Сохраняйте JSON конфигурации, результаты, Git commit и версии Python-зависимостей. Закреплённый инспектор показывает точную текущую конфигурацию." },
            { title: "CLI и Python API", body: "Те же компоненты можно создать из Python-словарей или запустить из сохранённого JSON эксперимента. Dry-run позволяет проверить развёрнутую сетку до вычислений." },
          ],
        },
        results: {
          title: "Файлы результатов",
          intro: "Локальный движок создаёт JSON и при необходимости CSV с траекторией. В статическом режиме можно исследовать файлы с компьютера.",
          topics: [
            { title: "JSON", body: "JSON содержит метаданные оптимизатора и среды, seeds, статус, итоговые метрики и, если включено, полную траекторию для проверки." },
            { title: "CSV и локальный анализ", body: "CSV подходит для внешних графиков и статистики. Загруженный файл читается браузером локально и не отправляется на GitHub Pages." },
          ],
        },
        analysis: {
          title: "Интерпретация эксперимента",
          intro: "Раздел появляется только после загрузки результата JSON или CSV.",
          topics: [
            { title: "Время и восстановление", body: "Проверяйте всю кривую, переходные процессы после изменения дрейфа и хвостовое окно. Один итоговый показатель может скрывать нестабильность." },
            { title: "Seeds и алгоритмы", body: "Сравнивайте распределения и неопределённость на одинаковых seeds. Траектории нужно интерпретировать в выбранной евклидовой или многообразной геометрии." },
          ],
        },
        gym: {
          title: "Адаптер Gymnasium",
          intro: "WindGymEnv представляет ту же динамическую задачу как RL-среду, не нарушая информационный барьер оракула.",
          topics: [
            { title: "Действия", body: "Absolute action напрямую предлагает следующую точку. Delta action задаёт ограниченное смещение. Проекция сохраняет допустимую геометрию." },
            { title: "Награды", body: "Negative regret измеряет избыточное значение функции; negative error — геометрическое расстояние до скрытого оптимума для оценки." },
            { title: "Ограниченная геометрия", body: "Simplex проецируется на вероятности. Stiefel использует ортонормированные фреймы, а Grassmann считает эквивалентными базисы, связанные ортогональным преобразованием." },
          ],
        },
      },
    },
    zh: {
      labels: { guide: "参考", contents: "页面指南", copy: "复制", source: "代码", use: "适用场景", caution: "注意", parameters: "系数说明", scenarios: "交互示例", collapse: "收起指南", expand: "展开指南" },
      pages: {
        overview: {
          title: "WIND 的工作方式",
          intro: "基准协议、软件架构与本地启动的简明指南。",
          topics: [
            { title: "基准目标", body: "WIND 比较优化算法在不同函数地形、漂移、噪声和信息约束下跟踪隐藏移动最优点的能力。", formula: String.raw`x_t\longrightarrow\mathcal O_t(x_t)\longrightarrow\theta_{t+1}` },
            { title: "交互协议", body: "在第 t 步，优化器先提交 xₜ，只获得预言机允许的观测，然后环境才更新。真实 θₜ 仅用于评估，不会暴露给算法。", bullets: ["不能预知未来环境状态", "严格区分一阶与零阶信息", "相同 seed 可复现漂移和噪声"] },
            { title: "软件架构", body: "core.py 定义环境；oracle.py 控制反馈；benchmark.py 执行评测；metrics.py 计算指标；experiment.py 包含参考优化器；gym_env.py 提供 RL 适配器。" },
            { title: "科学保证", body: "测试套件检查函数地形不变量、时间一致性、信息屏障、指标定义、结果导出和确定性 reset。", bullets: ["Observation 不暴露 θₜ", "Regret 使用匹配的环境状态", "结果元数据足以进行验证"] },
            { title: "下载后的本地启动", body: "在下载的仓库中打开 PowerShell 并执行以下命令。最后一条命令会在 http://127.0.0.1:8765 启动页面。", code: localRun },
          ],
          scenarios: true,
        },
        environment: { title: "环境参考", intro: "选择目标函数的几何结构和隐藏最优点的移动规律。", topics: [
          { title: "函数地形", body: "函数地形定义 fₜ(x)、可行几何和距离含义。选择卡片后会显示公式和所有可调系数。" },
          { title: "漂移", body: "漂移只在优化器行动后更新 θₜ。平稳、平滑、随机、跳变、自适应和流形漂移用于测试不同跟踪能力。" },
          { title: "初始状态与边界", body: "单个初值会广播到全部坐标；向量必须包含 dim 个元素。欧氏任务使用 box 边界，流形任务使用自身几何约束。" },
        ] },
        oracle: { title: "观测参考", intro: "预言机是优化器与隐藏环境之间唯一的信息通道。", topics: [
          { title: "反馈模式", body: "一阶模式提供梯度，零阶模式提供函数值，混合模式提供两者，scheduled 交替模式，offline replay 使用记录的 θ 轨迹。" },
          { title: "信息屏障", body: "Blind-value 一阶模式隐藏 fₜ(xₜ)。优化器不会获得 θₜ、干净噪声或未来状态。" },
          { title: "噪声通道", body: "函数值噪声与梯度噪声独立配置。它们改变观测，但不改变用于真实指标的干净函数地形。" },
        ] },
        optimizers: { title: "优化器参考", intro: "该目录展示参考实现，而不是排行榜。可以选择多个兼容方法并在相同环境与 seeds 下比较。", topics: [
          { title: "一阶方法", body: "这些方法需要梯度，包括 SGD、动量、自适应矩、镜像下降、dual averaging、近端更新和符号压缩。" },
          { title: "零阶方法", body: "这些方法只通过函数值估计方向。多阶段公式会分别展示扰动、测量、估计和更新。" },
          { title: "公平比较", body: "使用相同的 seeds、步数、环境和预言机策略，并同时考察跟踪质量与查询次数。" },
          { title: "自定义方法", body: "自定义方法实现 OptimizerProtocol、reset() 和 step(observation)，并声明所需的一阶或零阶反馈，无需继承基础优化器类。" },
        ] },
        runner: { title: "运行与指标", intro: "配置可复现实验、选择评估指标并导出完整实验说明。", topics: [
          { title: "Seeds 与重复", body: "每个优化器会对每个 seed 运行。多个 seeds 用于衡量波动，而不是只报告一次有利结果。", formula: String.raw`N_{\mathrm{runs}}=N_{\mathrm{optimizers}}N_{\mathrm{seeds}}` },
          { title: "指标", body: "Tracking error 衡量到 θₜ 的距离；dynamic regret 衡量额外损失；恢复与适应指标描述变化响应；query efficiency 考虑查询成本。" },
          { title: "可复现记录", body: "保留配置 JSON、结果文件、Git commit 以及 Python 依赖版本。固定检查器显示当前准确配置。" },
          { title: "CLI 与 Python API", body: "同一组件可以从 Python 字典构建，也可以从保存的实验 JSON 运行。使用 dry-run 可在计算前检查解析后的参数网格。" },
        ] },
        results: { title: "结果文件", intro: "本地引擎生成 JSON 和可选的长格式 CSV；静态模式可读取电脑上的文件。", topics: [
          { title: "JSON", body: "JSON 保存优化器与环境元数据、seeds、状态、最终指标以及可选的完整轨迹。" },
          { title: "CSV 与本地分析", body: "CSV 适合外部绘图和统计。所选文件只在浏览器本地读取，不会上传到 GitHub Pages。" },
        ] },
        analysis: { title: "实验解读", intro: "仅在加载 JSON 或 CSV 结果后显示分析页面。", topics: [
          { title: "时间与恢复", body: "检查完整曲线、漂移变化后的过渡过程和尾部窗口。单个最终数值可能掩盖不稳定性。" },
          { title: "Seeds 与算法", body: "在相同 seeds 上比较分布和不确定性，并根据所选欧氏或流形几何解释轨迹。" },
        ] },
        gym: { title: "Gymnasium 适配器", intro: "WindGymEnv 将相同动态优化任务表示为 RL 环境，同时保持预言机的信息屏障。", topics: [
          { title: "动作", body: "Absolute action 直接提出下一点；delta action 提出有界位移；几何投影保证动作可行。" },
          { title: "奖励", body: "Negative regret 衡量额外目标值；negative error 用几何距离评价隐藏最优点跟踪。" },
          { title: "约束几何", body: "Simplex 投影到概率分布；Stiefel 使用正交标架；Grassmann 将正交变换相关的基视为同一子空间。" },
        ] },
      },
    },
  };

  window.WIND_ENTITY_TEXT = {
    landscapes: {
      quadratic: { en: "A controlled convex baseline with an adjustable condition number.", ru: "Управляемая выпуклая базовая задача с настраиваемым числом обусловленности.", zh: "具有可调条件数的凸基准问题。" },
      pnorm: { en: "Changes the local geometry through the p-norm and anisotropic conditioning.", ru: "Меняет локальную геометрию через p-норму и анизотропную обусловленность.", zh: "通过 p 范数和各向异性条件数改变局部几何。" },
      rosenbrock: { en: "A curved non-convex valley that tests direction coupling.", ru: "Изогнутая невыпуклая долина для проверки связанности направлений.", zh: "用于测试方向耦合的弯曲非凸谷。" },
      multiextremal: { en: "Multiple local basins test exploration and sensitivity to initialization.", ru: "Несколько локальных бассейнов проверяют исследование и чувствительность к инициализации.", zh: "多个局部盆地测试探索能力和对初始化的敏感性。" },
      robust: { en: "Huber-like growth limits the influence of large coordinate errors.", ru: "Huber-подобный рост ограничивает влияние больших координатных ошибок.", zh: "类似 Huber 的增长限制大坐标误差的影响。" },
      simplex: { en: "Optimization over non-negative coordinates that sum to one.", ru: "Оптимизация по неотрицательным координатам с суммой, равной единице.", zh: "在非负且总和为一的坐标上优化。" },
      stiefel: { en: "Tracks an ordered orthonormal frame on the Stiefel manifold.", ru: "Отслеживает упорядоченный ортонормированный фрейм на многообразии Штифеля.", zh: "在 Stiefel 流形上跟踪有序正交标架。" },
      grassmann: { en: "Tracks a subspace independently of the chosen orthonormal basis.", ru: "Отслеживает подпространство независимо от выбранного ортонормированного базиса.", zh: "独立于具体正交基跟踪子空间。" },
    },
    drifts: {
      stationary: { en: "A fixed optimum isolates optimizer convergence.", ru: "Неподвижный оптимум изолирует сходимость оптимизатора.", zh: "固定最优点用于隔离优化器收敛性质。" },
      linear: { en: "Constant velocity measures steady-state tracking lag.", ru: "Постоянная скорость измеряет установившееся запаздывание.", zh: "恒定速度用于衡量稳态跟踪滞后。" },
      random_walk: { en: "Stochastic increments test continual adaptation under uncertainty.", ru: "Случайные приращения проверяют постоянную адаптацию в условиях неопределённости.", zh: "随机增量测试不确定性下的持续适应。" },
      cyclic: { en: "Periodic motion exposes phase lag and resonance.", ru: "Периодическое движение выявляет фазовое запаздывание и резонанс.", zh: "周期运动揭示相位滞后与共振。" },
      jump: { en: "Abrupt changes are used for recovery-time metrics.", ru: "Резкие изменения используются для метрик времени восстановления.", zh: "突变用于评估恢复时间。" },
      adaptive: { en: "The optimum reacts to the optimizer in pursuit or evasion mode.", ru: "Оптимум реагирует на алгоритм в режиме pursuit или evasion.", zh: "最优点以追逐或规避模式响应优化器。" },
      sparse: { en: "Only a subset of coordinates moves at each step.", ru: "На каждом шаге движется только часть координат.", zh: "每一步仅有部分坐标移动。" },
      stiefel: { en: "A manifold-preserving drift for orthonormal targets.", ru: "Дрейф, сохраняющий многообразие ортонормированных целей.", zh: "保持正交目标流形结构的漂移。" },
    },
    oracles: {
      "first-order": { en: "Gradient feedback; blind-value mode can hide the scalar loss.", ru: "Градиентная обратная связь; blind-value может скрывать значение функции.", zh: "提供梯度反馈；blind-value 可隐藏标量损失。" },
      "zero-order": { en: "Function values only; suitable for derivative-free methods.", ru: "Только значения функции; подходит для безградиентных методов.", zh: "仅提供函数值，适用于无导数方法。" },
      hybrid: { en: "Value and gradient observations with independent noise channels.", ru: "Значение и градиент с независимыми каналами шума.", zh: "同时提供函数值和梯度，并使用独立噪声通道。" },
      scheduled: { en: "Alternates first- and zero-order access according to a schedule.", ru: "Чередует first-order и zero-order доступ по расписанию.", zh: "按时间表交替使用一阶和零阶反馈。" },
      offline: { en: "Replays a recorded optimum trajectory for paired comparisons.", ru: "Воспроизводит записанную траекторию оптимума для парных сравнений.", zh: "重放已记录的最优点轨迹以进行配对比较。" },
    },
    noises: {
      none: { en: "No observation corruption.", ru: "Наблюдение без искажений.", zh: "不添加观测噪声。" },
      gaussian: { en: "Independent light-tailed additive noise controlled by σ.", ru: "Независимый аддитивный гауссовский шум с масштабом σ.", zh: "由 σ 控制的独立高斯加性噪声。" },
      heavy_tailed: { en: "Rare large errors controlled by tail exponent α and scale.", ru: "Редкие большие ошибки с показателем хвоста α и scale.", zh: "由尾指数 α 和 scale 控制的稀有大误差。" },
      correlated: { en: "AR(1)-style temporal dependence controlled by φ.", ru: "Временная зависимость типа AR(1), задаваемая φ.", zh: "由 φ 控制的 AR(1) 时间相关噪声。" },
      quantized: { en: "Rounds observations to a grid with spacing δ.", ru: "Округляет наблюдения к сетке с шагом δ.", zh: "将观测量化到间隔为 δ 的网格。" },
      multiplicative: { en: "Noise magnitude scales with the observed signal.", ru: "Масштаб шума зависит от величины наблюдаемого сигнала.", zh: "噪声幅度随观测信号大小变化。" },
      sparse: { en: "Corrupts only a Bernoulli-selected fraction p of observations.", ru: "Искажает только выбранную по Bernoulli долю p наблюдений.", zh: "仅以 Bernoulli 概率 p 扰动部分观测。" },
    },
  };

  const optimizerDescriptions = {
    SGD: ["Gradient descent with optional momentum.", "Градиентный спуск с необязательным импульсом.", "带可选动量的梯度下降。"],
    SGD_Polyak: ["Averages SGD iterates to reduce variance.", "Усредняет итерации SGD для снижения дисперсии.", "通过平均 SGD 迭代降低方差。"],
    HeavyBall: ["Uses persistent velocity to accelerate smooth directions.", "Использует инерционную скорость для ускорения по гладким направлениям.", "使用持续速度加速平滑方向。"],
    Nesterov: ["Evaluates the descent direction at a look-ahead point.", "Оценивает направление спуска в упреждающей точке.", "在前瞻点估计下降方向。"],
    Adam: ["Combines first and second gradient moments with bias correction.", "Сочетает первый и второй моменты градиента с коррекцией смещения.", "结合一阶和二阶梯度矩并进行偏差修正。"],
    AdamW: ["Adam with an explicit weight-decay contribution.", "Adam с явным вкладом weight decay.", "带显式权重衰减的 Adam。"],
    AMSGrad: ["Uses a non-decreasing second-moment maximum for stability.", "Использует неубывающий максимум второго момента для устойчивости.", "使用单调不减的二阶矩最大值增强稳定性。"],
    SMD: ["Entropy mirror descent that naturally preserves the simplex.", "Энтропийный зеркальный спуск, сохраняющий симплекс.", "自然保持单纯形的熵镜像下降。"],
    RDA: ["Accumulates gradients before an L1 soft-threshold update.", "Накапливает градиенты перед L1 soft-threshold обновлением.", "累积梯度后执行 L1 软阈值更新。"],
    ProxSGD: ["Separates a gradient step from an L1 proximal step.", "Разделяет градиентный и L1 proximal шаги.", "将梯度步骤与 L1 近端步骤分离。"],
    AdaptiveLR: ["Shrinks the learning rate as the gradient norm grows.", "Уменьшает шаг при росте нормы градиента.", "随梯度范数增大而缩小学习率。"],
    SignSGD: ["Uses only coordinate-wise gradient signs.", "Использует только покоординатные знаки градиента.", "仅使用逐坐标梯度符号。"],
    RandomSearch: ["Samples around the best point observed so far.", "Сэмплирует вокруг лучшей найденной точки.", "围绕当前最佳点进行采样。"],
    OnePointSPSA: ["Builds a direction estimate from one perturbed measurement.", "Строит оценку направления по одному возмущённому измерению.", "通过一次扰动测量估计方向。"],
    FiniteDiffCentral: ["Uses symmetric coordinate queries for a finite-difference gradient.", "Использует симметричные координатные запросы для конечной разности.", "使用对称坐标查询估计有限差分梯度。"],
    FDSA: ["Estimates a gradient along a random finite-difference direction.", "Оценивает градиент вдоль случайного конечно-разностного направления.", "沿随机有限差分方向估计梯度。"],
    SPSA: ["Uses simultaneous random perturbations and paired measurements.", "Использует одновременные случайные возмущения и парные измерения.", "使用同时随机扰动和成对测量。"],
    ZOSGD: ["Applies SGD to a Gaussian-smoothed gradient estimate.", "Применяет SGD к оценке градиента сглаженной функции.", "对高斯平滑梯度估计应用 SGD。"],
    ZOSignSGD: ["Uses the sign of a zero-order directional estimate.", "Использует знак zero-order оценки направления.", "使用零阶方向估计的符号。"],
    QuadraticInterpolation: ["Fits a one-dimensional quadratic along a random direction.", "Строит одномерную квадратичную модель вдоль случайного направления.", "沿随机方向拟合一维二次模型。"],
    KieferWolfowitz: ["Classical finite-difference stochastic approximation with decaying scales.", "Классическая конечно-разностная стохастическая аппроксимация с убывающими масштабами.", "具有衰减尺度的经典有限差分随机逼近。"],
    NedicSubgradient: ["Uses a random directional subgradient and diminishing step.", "Использует случайный направленный субградиент и убывающий шаг.", "使用随机方向次梯度和递减步长。"],
    AcceleratedSPSA: ["Adds momentum to the SPSA gradient estimate.", "Добавляет импульс к оценке градиента SPSA.", "为 SPSA 梯度估计添加动量。"],
    CMAES: ["Updates a sampling distribution from the best population members.", "Обновляет распределение сэмплирования по лучшим членам популяции.", "根据种群中的优秀成员更新采样分布。"],
    GPUCB: ["Balances an observed direction with an uncertainty bonus.", "Балансирует наблюдаемое направление и бонус неопределённости.", "在观测方向与不确定性奖励之间权衡。"],
  };
  window.WIND_OPTIMIZER_TEXT = Object.fromEntries(Object.entries(optimizerDescriptions).map(([name, values]) => [name, { en: values[0], ru: values[1], zh: values[2] }]));

  window.WIND_PARAMETER_HELP = {
    en: { lr: "learning rate", lr0: "base learning rate", momentum: "velocity weight", beta: "momentum coefficient", beta1: "first-moment decay", beta2: "second-moment decay", eps: "numerical stabilizer", weight_decay: "L2 decay", lambda_reg: "L1 regularization", scale: "search radius", perturb: "probe displacement", h: "finite-difference spacing", mu: "smoothing radius", cn: "initial perturbation scale", sigma: "sampling spread", population_size: "population size; 0 selects automatically" },
    ru: { lr: "шаг обучения", lr0: "базовый шаг", momentum: "вес скорости", beta: "коэффициент импульса", beta1: "затухание первого момента", beta2: "затухание второго момента", eps: "численная стабилизация", weight_decay: "L2-регуляризация", lambda_reg: "L1-регуляризация", scale: "радиус поиска", perturb: "величина пробного возмущения", h: "шаг конечной разности", mu: "радиус сглаживания", cn: "начальный масштаб возмущения", sigma: "разброс сэмплирования", population_size: "размер популяции; 0 выбирается автоматически" },
    zh: { lr: "学习率", lr0: "基础学习率", momentum: "速度权重", beta: "动量系数", beta1: "一阶矩衰减", beta2: "二阶矩衰减", eps: "数值稳定项", weight_decay: "L2 衰减", lambda_reg: "L1 正则化", scale: "搜索半径", perturb: "探测扰动幅度", h: "有限差分间隔", mu: "平滑半径", cn: "初始扰动尺度", sigma: "采样分布宽度", population_size: "种群大小；0 表示自动选择" },
  };

  window.WIND_SCENARIOS = [
    { id: "smooth", labels: { en: "Smooth tracking", ru: "Плавное отслеживание", zh: "平滑跟踪" }, descriptions: { en: "Quadratic + linear drift + SGD", ru: "Quadratic + linear drift + SGD", zh: "Quadratic + 线性漂移 + SGD" }, landscape: "quadratic", drift: "linear", oracle: "first-order", optimizer: "SGD", valueNoise: "gaussian", gradNoise: "gaussian" },
    { id: "noisy-zo", labels: { en: "Noisy zero-order", ru: "Шумный zero-order", zh: "含噪零阶" }, descriptions: { en: "Random walk + value noise + SPSA", ru: "Random walk + шум значения + SPSA", zh: "随机游走 + 函数值噪声 + SPSA" }, landscape: "quadratic", drift: "random_walk", oracle: "zero-order", optimizer: "SPSA", valueNoise: "gaussian", gradNoise: "none" },
    { id: "simplex", labels: { en: "Simplex allocation", ru: "Распределение на симплексе", zh: "单纯形分配" }, descriptions: { en: "Simplex + cyclic drift + mirror descent", ru: "Simplex + cyclic drift + mirror descent", zh: "Simplex + 周期漂移 + 镜像下降" }, landscape: "simplex", drift: "cyclic", oracle: "first-order", optimizer: "SMD", valueNoise: "gaussian", gradNoise: "gaussian" },
  ];
})();
