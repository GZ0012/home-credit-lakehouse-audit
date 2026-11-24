# Volkova silver 层特征说明

本文说明 `silver_volkova` schema 下每张表的特征怎么生成、怎么聚合，以及为什么这样做。

依据是 Volkova 方案在 commit [`d5e228c`](https://github.com/evgeniavolkova/kagglehomecredit2024/tree/d5e228cdf7e9bf81d10805cff205b473a94c2063) 的源码：

- `homecredit/data/data_processor_2.py`：处理逻辑
- `features/features_2.csv`：每个字段用哪些聚合函数，以及是否保留（`include`）

**关于"原因"：** 作者的代码和配置只规定了怎么做，没有写为什么。本文的"原因"分两种：

- **源码可证**：能从代码直接看出来的，比如某个结果被覆盖、某个配置项的作用。
- **推断**：根据字段含义做的解释，只用于帮助理解，不代表作者的原意。

| 表 | 脚本 | 粒度 | 是否聚合 |
|---|---|---|---|
| `train_case_sources` | `01_case_sources.py` | 每个 `case_id` 一行 | 否，只做左连接和补值 |
| `train_applprev_1_features` | `02_applprev_1.py` | 每个 `case_id` 一行 | 是，历史申请按案例聚合 |
| `train_applprev_2_features` | `03_applprev_2.py` | 每个 `case_id` 一行 | 是，历史申请的联系人和卡片记录按案例聚合 |
| `train_person_features` | `04_person.py` | 每个 `case_id` 一行 | 部分：申请人一行原样保留，关联人和 person2 按案例聚合 |
| `train_credit_bureau_1_features` | `05_credit_bureau_1.py` | 每个 `case_id` 一行 | 是，征信合同拆成 active 和 closed 两种视图后按案例聚合 |
| `train_static_0_features` | `06_static_0.py` | 每个 `case_id` 一行 | 否，本身就是案例级，只做补值和公式 |
| `train_static_cb_0_features` | `07_static_cb_0.py` | 每个 `case_id` 一行 | 否，本身就是案例级，只做补值和一个比率 |
| `train_credit_bureau_2_features` | `08_credit_bureau_2.py` | 每个 `case_id` 一行 | 是，征信每月还款记录拆成 active 和 closed 后按案例聚合 |
| `train_debitcard_deposit_other_features` | `09_debitcard_deposit_other.py` | 每个 `case_id` 一行 | 是，借记卡、存款、其他账户分别按案例聚合 |
| `train_tax_registry_features` | `10_tax_registry.py` | 每个 `case_id` 一行 | 是，三个税务源分别按案例聚合 |
| `gold_volkova.train_features` | `gold/volkova/01_train_features.py` | 每个 `case_id` 一行 | 否，合并以上所有表，再做跨源处理（gold 层） |

---

## 1. `train_case_sources`

### 1.1 来源和连接方式

以 `train_base` 为主表，依次 LEFT JOIN 三个案例级来源。这三个来源本身就是每个案例一行，所以**不需要聚合**。

| 前缀 | 来源表 | 取哪些行 | 原因 |
|---|---|---|---|
| — | `train_base` | 全部 | 带标签的案例总体，决定输出的行数 |
| `person1_` | `train_person_1` | 仅 `num_group1 = 0` | `num_group1 = 0` 是申请人本人，其他行是关联人。作者 `_handle_person1` 的拆分方式（源码可证） |
| `static0_` | `train_static_0` | 全部 | 本来就是每个案例一行 |
| `staticcb0_` | `train_static_cb_0` | 全部 | 本来就是每个案例一行，数据来自外部征信 |

每个来源都会加一个 `has_<前缀>` 标记，表示这个案例有没有匹配到该来源的行。

脚本在连接前后都会检查两件事：`case_id` 唯一，总行数和 `train_base` 一致。

### 1.2 补值（COALESCE）规则

同一个信息在原始数据里常常有好几列，比如新旧版本的字段、不同渠道的字段。作者的做法是：按优先级用 `COALESCE` 合成一列，主字段在前。合并之前，分类字段里的掩码值 `a55475b1` 先置为 null，所以掩码值不会挡住后面的备选值。

| 派生列 | 作者列名 | 优先级（从左到右） | 分类字段 | 说明 |
|---|---|---|---|---|
| `person_birthdate` | `birth_259D_1_person1` | `birth_259D` → `birthdate_87D` | | |
| `person_gender` | `gender_992L_1_person1` | `gender_992L` → `sex_738L` | ✓ | |
| `person_relationship` | `relationshiptoclient_415T_1_person1` | `relationshiptoclient_415T` → `relationshiptoclient_642T` | ✓ | **申请人这一行的结果没有进入模型**：`relationshiptoclient_415T` 的 `include = 0`，`_handle_person1` 把它从申请人行删掉了（源码可证）。这个补值只影响关联人的 `mode`，见第 4 节。01 的报表里这条标记为 `used_by_model = false` |
| `external_birthdate` | `birthdate_574D_2_staticcb0` | `birthdate_574D` → `dateofbirth_337D` → `dateofbirth_342D` | | |
| `external_education` | `education_1103M_2_staticcb0` | `education_1103M` → `education_88M` | ✓ | |
| `external_marital_status` | `maritalst_385M_2_staticcb0` | `maritalst_385M` → `maritalst_893M` | ✓ | |
| `external_tax_mean` | `pmtaverage_3A_2_staticcb0` | `pmtaverage_4527227A` → `pmtaverage_4955615A` → `pmtaverage_3A` | | |
| `external_tax_count` | `pmtcount_4527229L_2_staticcb0` | `pmtcount_4527229L` → `pmtcount_4955617L` → `pmtcount_693L` | | |
| `external_tax_assignment_date` | `assignmentdate_238D_2_staticcb0` | `assignmentdate_238D` → `assignmentdate_4527235D` → `assignmentdate_4955616D` | | |
| `outstanding_balance` | `sumoutstandtotal_3546847A_2_static0` | `sumoutstandtotal_3546847A` → `sumoutstandtotalest_4493215A` | | **模型没有用到**：源码随后用同名列覆盖了它（源码可证），这里只为审计保留 |
| `applicant_birthdate` | `birthdate_1_final0` | `external_birthdate` → `person_birthdate` | | 优先用外部征信的生日，没有再用申请表上的 |
| `applicant_main_income` | `mainoccupationinc_1_final0` | `static0_maininc_215A` → `person1_mainoccupationinc_384A` | | |

**为什么保留原始输入列：** 为了审计。每个派生值都要能追溯到具体来源，包括被掩码的原始分类值。补值效果的统计（补值前缺失多少、补回多少、补值后还缺多少）写在 `reports/01_case_sources/fallback_missingness.csv`。

**日期处理：** 输出表里日期列保留为日历日期，只有 `birthdate_1_final0` 另外换算成相对 `date_decision` 的天数（负数表示过去）。作者是先把日期换算成天数再合并；所有输入都相对同一个 `date_decision`，所以两种做法结果一样。

---

## 2. `train_applprev_1_features`

### 2.1 总流程

`train_applprev_1` 是历史申请明细，一个案例对应多条历史申请。处理分三步：

0. **分片规则。** `train_applprev_1` 在 Kaggle 里分成两个分片文件。作者逐个文件聚合，再对每个案例保留第一个结果（`unique(keep="first")`，源码可证），所以一个案例如果跨了两个分片，只用第一个分片里的行。脚本按 `_source_file` 复现这条规则。
1. **先在明细上算行级特征。** 先 INNER JOIN `train_case_sources`，取 `date_decision`、申请人收入和当前贷款额，然后计算比率、日期差等派生列。
2. **再聚合成每个案例一行。** 分四组聚合，另外单独取最近一次申请那一行。
3. **最后 LEFT JOIN 到全部案例。** 没有历史申请的案例，计数列补 0，其他列为 null。

**为什么先算行级特征再聚合（推断）：** "每笔贷款的月供 / 收入"取平均，和"平均月供 / 平均收入"不是一回事。比率必须在单笔申请上先算出来，再看它在历史上的分布。

### 2.2 行级预处理

| 处理 | 做法 | 原因 |
|---|---|---|
| 日期 | 所有 `*D` 列换算成 `日期 − date_decision` 的天数 | 作者 `_handle_dates` 的做法（源码可证）。换算后日期变成数值，可以求 min、max、mean，也不受申请时间不同的影响 |
| 掩码分类 | 所有字符串列里的 `a55475b1` 置为 null | 作者 `handle_cat` 的做法（源码可证）。这个值表示缺失，不是一个真实的类别 |
| 0/1 标志 | `isbid*`、`isdebit*` 转成整数 0/1 | 作者 `set_dtypes` 里转成了 `Int8`（源码可证）。转成整数后才能求 min/max，也才能进入 approved/refused 组的数值聚合 |
| 金额、DPD | `*A`、`*P` 转成 double | 作者统一转成 `Float64`（源码可证） |

### 2.3 行级派生列

派生列没有单独的聚合配置，**沿用所属原字段的聚合函数**。比如 `annuity_853A_2` 用的就是 `annuity_853A` 的配置（源码可证：`"_".join(col.split("_")[:-1])`）。

| 派生列 | 公式 | 沿用哪个字段的聚合 | 含义（推断） |
|---|---|---|---|
| `mainoccupationinc_437A_2` | `COALESCE(历史收入, person1 当前收入)` | `mainoccupationinc_437A`：min, max | 历史收入缺失时用当前收入顶上 |
| `currdebt_94A_2` | 当前债务 / 历史收入 | `currdebt_94A`：min, max | 债务收入比 |
| `annuity_853A_2` | 月供 / 历史收入 | `annuity_853A`：min, max, rangefirstlast, mean | 月供收入比，即还款负担 |
| `outstandingdebt_522A_2` | 未偿债务 / 历史收入 | `outstandingdebt_522A`：max, min, mean | 未偿债务收入比 |
| `credamount_590A_2` | 贷款额 / 历史收入 | `credamount_590A`：min, max, mean | 借款规模相对收入 |
| `annuity_853A_3` | 月供 / 贷款额 | 同 `annuity_853A` | 每期还款占本金的比例 |
| `annuity_853A_4` | (月供 × 期数 / 贷款额 − 1) / 期数 | 同 `annuity_853A` | 平均到每期的隐含利率 |
| `downpmt_134A_2` | 首付 / 贷款额 | `downpmt_134A`：min, max, mean | 首付比例 |
| `credamount_590A_3` | 历史贷款额 / 当前贷款额（static0） | 同 `credamount_590A` | 这次借的比以前多还是少 |
| `credacc_credlmt_575A_2` | 信用额度 / 实际余额 | `credacc_credlmt_575A`：min, max | 额度使用情况 |
| `dtlastpmt_581D_2` | 最后付款日 − 首期分期日 | `dtlastpmt_581D`：min, max, mean | 实际还款持续了多久 |
| `dtlastpmt_581D_3` | 最后付款日 − 申请创建日 | 同 `dtlastpmt_581D` | 从申请到最后一次付款的间隔 |

**注意：**

- **比率的分母用原始历史收入，不用补值后的 `_2` 列**（源码可证）。所以收入补值只体现在 `mainoccupationinc_437A_2` 这一列本身，不影响任何比率。
- **除以零**：polars 会得到 ±inf，参与聚合后再在最后一步统一置为 null。Spark 默认返回 null，开 ANSI 时会直接报错，所以脚本手动生成 inf，保证 max、mean 等结果和作者一致。
- **`dtlastpmt` 的补值**：作者先用 `dtlastpmtallstes_3545839D` 补 `dtlastpmt_581D`，但紧接着又用上表中的日期差覆盖了同名列（源码可证），所以这个补值不进模型。脚本不保存这一列，只在 `fallback_coverage.csv` 里报告它的覆盖率，并标记 `used_by_model = false`。

### 2.4 聚合函数定义

| 函数 | 定义 | 对 null 的处理 |
|---|---|---|
| `min` / `max` | 最小值 / 最大值 | 忽略 null |
| `mean` | 平均值 | 忽略 null |
| `count` | 非空值个数 | 只数非空值 |
| `range` | `max − min` | 忽略 null |
| `first` | 按时间顺序最早的一个非空值 | 跳过 null |
| `rangefirstlast` | 最早的非空值 − 最晚的非空值 | 跳过 null |
| `nunique` | 不同取值的个数 | **null 也算作一个值**（与 polars 的 `n_unique` 一致） |
| `nuniquetotal` | `nunique / 非空值个数` | 衡量这个字段的取值有多分散 |
| `mode` | 出现次数最多的值 | 忽略 null；平局时取最小值，作者没有规定平局怎么处理 |

**时间顺序：** 在每个案例内按 `creationdate_885D` 升序排列，null 排最前（作者对 `creationdate_885D_1` 做了 `sort`，源码可证）。同一天有多条时按 `num_group1` 排，这一点作者没有规定。

### 2.5 分组

| 分组 | 条件 | 聚合哪些列 | 列名后缀 | 原因 |
|---|---|---|---|---|
| 全历史 | 不过滤 | 全部列 | 无 | 客户全部申请历史的整体画像 |
| approved | `approvaldate_319D` 不为空 | **仅数值列** | `_approved` | 拿到贷款的申请反映实际的借贷和还款行为（推断） |
| refused | `approvaldate_319D` 为空 | **仅数值列** | `_refused` | 被拒或取消的申请反映客户的申请倾向和被拒记录（推断） |
| 近 5 年 | `creationdate` 天数 ≥ −1825 | 全部列 | `_creationdate_885D_5y` | 更早的历史和当前风险关系更弱，单独看近期（推断）。窗口用 `creationdate`，因为它在配置里是唯一的 `date_col = 1` |
| 最近一次申请 | `num_group1 = 0` | **不聚合**，原样保留这一行 | 无 | 作者称之为 "last application"，用作最新状态的快照。注意：源码没有证明 `num_group1 = 0` 一定是时间上最近的一次 |

每组还会生成一列 `case_id_1_count_<后缀>`，记录这一组里的历史条数。最后没有历史的案例，计数补 0（源码可证：作者对含 `case_id_` 的列 `fill_null(0)`）。

**approved/refused 为什么只聚合数值列：** 作者在这两组前先做了 `select(NUMERIC_DTYPES)`（源码可证），所以字符串类的分类字段不参与这两组。

### 2.6 逐字段聚合方法

以下"聚合"列来自 `features_2.csv`。`include = 0` 的字段仍然参与各组聚合，只是不保留在"最近一次申请"那一行里。

**金额和余额**

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `annuity_853A` | 月供 | min, max, mean, rangefirstlast | 水平和区间，再加上从最早到最晚的变化趋势 |
| `credamount_590A` | 贷款额或卡额度 | min, max, mean | 借款规模的分布 |
| `currdebt_94A` | 当前债务 | min, max | 债务的极值 |
| `outstandingdebt_522A` | 未偿债务 | max, min, mean | 未偿债务的分布 |
| `downpmt_134A` | 首付 | min, max, mean | 首付能力 |
| `mainoccupationinc_437A` | 历史申请里填的收入 | min, max | 收入区间 |
| `credacc_actualbalance_314A` | 信用账户实际余额 | min, max | 余额极值 |
| `credacc_credlmt_575A` | 信用卡额度 | min, max | 额度极值 |
| `credacc_maxhisbal_375A` | 历史最高余额 | min, max | 极值 |
| `credacc_minhisbal_90A` | 历史最低余额 | min, max, mean | 极值和平均水平 |
| `revolvingaccount_394A` | 循环账户 | max | `include = 0`，仅聚合 |

**逾期**

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `actualdpd_943P` | 实际逾期天数 | min, max | max 是历史上最严重的逾期，风险信号最强；min 表示是否有过从不逾期的合同 |
| `maxdpdtolerance_577P` | 带宽限期的最大逾期天数 | min, max | 同上 |

**日期（已换算成相对 `date_decision` 的天数）**

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `creationdate_885D` | 申请创建日 | min, max, mean, range | min 是最早一次申请，反映历史有多长；max 是最近一次，反映离现在多近；range 是活跃跨度。同时也是"近 5 年"窗口的依据 |
| `approvaldate_319D` | 审批日 | min, max, range, count | count 等于获批次数（审批日非空即获批） |
| `dateactivated_425D` | 合同激活日 | min, max | 最早和最近的激活时间 |
| `dtlastpmt_581D` | 最后付款日 | min, max, mean | 还款活动离现在多近 |
| `dtlastpmtallstes_3545839D` | 最后付款日（另一来源） | min, max | 同上 |
| `firstnonzeroinstldate_307D` | 首个非零分期日 | min, max, mean | 还款开始的时间 |
| `employedfrom_700D` | 就业开始日 | min, max, mean | 工作的稳定性 |

**计数、期数和标志**

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `tenor_203L` | 分期期数 | min, max | 期限的偏好 |
| `pmtnum_8L` | 已还期数 | min, max | 还款进度 |
| `credacc_transactions_402L` | 信用账户交易次数 | min, max | 账户活跃度 |
| `childnum_21L` | 子女数 | first, max | 子女数一般只增不减。first 是最早的记录，max 是最新的上限 |
| `isbidproduct_390L` | 是否交叉销售 | min, max | 0/1 标志：max = 1 表示有过，min = 1 表示每次都是 |
| `isdebitcard_527L` | 是否借记卡产品 | min, max | 同上 |
| `byoccupationinc_3656910L` | 历史申请中的收入 | min, max, mean, rangefirstlast | `include = 0`，仅聚合 |

**分类字段**

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `status_219L` | 申请状态 | nunique, nuniquetotal, mode | 出现过几种状态，以及最常见的状态 |
| `credtype_587L` | 信贷类型 | nunique, nuniquetotal, mode | 产品是否多样，以及主要产品 |
| `credacc_status_367L` | 账户状态 | nunique, nuniquetotal, mode | 同上 |
| `familystate_726L` | 婚姻或家庭状态 | nunique, nuniquetotal, mode | 状态是否发生过变化 |
| `inittransactioncode_279L` | 初始交易类型 | nunique, nuniquetotal, mode | 同上 |
| `cancelreason_3545846M` | 取消原因 | nunique, nuniquetotal, mode | 主要的取消原因 |
| `rejectreason_755M` | 拒绝原因 | nunique, nuniquetotal, mode | 主要的拒绝原因 |
| `rejectreasonclient_4145042M` | 客户侧拒绝原因 | nunique, nuniquetotal, mode | 同上 |
| `district_544M` | 地址所在区 | nunique, nuniquetotal | 取值很多，众数意义不大，所以只看换过几次地址 |
| `profession_152M` | 职业 | nunique, nuniquetotal | 取值很多，同上，只看职业变化次数 |
| `education_1138M` | 教育程度 | mode | 基本不变，取最常见的值 |
| `postype_4733339M` | 销售点类型 | mode | 主要的申请渠道 |

### 2.7 输出列命名

```
<原字段>_<行级序号>_<聚合函数><分组后缀>_applprev1
```

例子：

- `annuity_853A_1_min_applprev1`：全历史的月供最小值
- `annuity_853A_2_mean_approved_applprev1`：已批准申请里月供收入比的平均值
- `status_219L_1_mode_creationdate_885D_5y_applprev1`：近 5 年最常见的申请状态
- `annuity_853A_1_applprev1`：最近一次申请（`num_group1 = 0`）的月供，没有聚合
- `case_id_1_count__refused_applprev1`：被拒申请的条数。双下划线是作者原来的命名，保留以便对照源码

另外几列：

| 列 | 说明 |
|---|---|
| `has_applprev1` | 这个案例是否有历史申请 |

`employed_length_1_final0` = `COALESCE(employedfrom_700D_1_applprev1, empl_employedfrom_271D_1_person1)`：优先用最近一次申请里的就业开始日，没有再用申请人本人的（源码可证，作者在 `FinalDataProcessor` 中做）。这一列在 gold 01 里计算，本表不再输出；`fallback_coverage.csv` 里的 `employment_start` 一行只统计有多少案例需要用到备选值。

整张表大约 530 列，确切数量取决于 bronze 里各个 `L` 字段的类型，因为字符串类型不进 approved/refused 组。

### 2.8 最后一步的数值处理

| 处理 | 原因 |
|---|---|
| ±inf 置为 null | 作者最后一步的做法（源码可证）。inf 来自除以零 |
| NaN 置为 null | 0 / 0 会得到 NaN。作者保留 NaN，模型也把它当缺失处理，两种写法效果一样 |
| 计数列的 null 补 0 | 没有历史的案例，条数就是 0，不是未知 |
| 原始金额、正常的 0、负的日期偏移量 | 都保留。它们是真实的取值，不是缺失 |

报表写在 `reports/02_applprev_1/`：`row_coverage.csv` 记录行数和案例覆盖，`fallback_coverage.csv` 记录各补值的效果。

---

## 3. `train_applprev_2_features`

### 3.1 总流程

`train_applprev_2` 是深度 2 的明细：每笔历史申请（`num_group1`）下面可以有多条联系人或卡片记录（`num_group2`）。

作者**跳过历史申请这一层，把所有记录直接聚合到 `case_id`**（源码可证：`group_by(COL_ID)`），没有先按每笔申请汇总一次。

1. INNER JOIN `train_case_sources`，只用来限定案例范围，不需要案例级输入。
2. 所有字符串列里的 `a55475b1` 置为 null。
3. `_process_applprev2` 只把列名改成 `_1`，**没有行级公式**（源码可证）。
4. 只聚合一组，即全部记录。没有 approved/refused 分组，没有时间窗口，也没有"最近一次"那一行（源码可证）：approved/refused 和"最近一次"只在 applprev1 里做；时间窗口需要 `date_col`，而 applprev2 没有日期列。
5. LEFT JOIN 到全部案例，没有记录的案例计数补 0。

### 3.2 逐字段聚合方法

三个字段都是分类字段，聚合方法相同：`nunique, nuniquetotal, mode, first`。

| 字段 | 含义 | 原因（推断） |
|---|---|---|
| `cacccardblochreas_147M` | 卡片冻结原因 | 有过几种冻结原因，最常见的是哪种，最早记录的是哪种 |
| `conts_type_509L` | 联系人类型 | 留过几种联系方式，主要的是哪种 |
| `credacc_cards_status_52L` | 卡片状态 | 卡片状态是否多样，以及主要状态 |

**为什么只用分类聚合（推断）：** 这三个字段都没有数值含义，所以只能看"有几种"（nunique）、"取值多分散"（nuniquetotal）、"最常见的"（mode）和"最早的"（first）。

### 3.3 `first` 的顺序问题

作者对 applprev2 **没有排序**（源码可证：`col_sort` 只给 applprev1 和三个税务源设了值），所以 `first` 取的是原始文件里第一个非空值。

Spark 表没有固定的行顺序，脚本用 (`num_group1`, `num_group2`) 升序来代替文件顺序。这依赖一个**未经验证的假设**：Kaggle 原始文件就是按这个顺序存的。

只有一个字段在同一案例里有不止一种非空值时，`first` 才会受顺序影响，其余案例无论按什么顺序结果都一样。

### 3.4 输出列

共 3 × 4 = 12 个聚合列，另有：

| 列 | 说明 |
|---|---|
| `case_id_1_count__applprev2` | 这个案例在 applprev2 里的记录条数，没有记录为 0 |
| `has_applprev2` | 是否有 applprev2 记录 |

例子：`conts_type_509L_1_mode_applprev2` 是这个客户所有历史申请里最常见的联系人类型。

`nuniquetotal` 在全部值都为空时会出现 0 做除数，结果按作者的做法置为 null。

报表写在 `reports/03_applprev_2/`：`row_coverage.csv` 记录行数和案例覆盖。

---

## 4. `train_person_features`

这张表合并了两个源，列名后缀分别是 `_person1` 和 `_person2`。

| 源 | 一行代表什么 | 怎么处理 |
|---|---|---|
| `train_person_1`，`num_group1 = 0` | 申请人本人 | 原样保留，不聚合 |
| `train_person_1`，`num_group1 ≠ 0` | 关联人（配偶、联系人等） | 按案例聚合 |
| `train_person_2` | 每个人的地址、雇主等，深度 2 | 按案例聚合 |

### 4.1 person1 的行级处理

顺序和作者一致（源码可证）：

1. INNER JOIN `train_case_sources`，取 `date_decision`。
2. 日期换算成天数，`a55475b1` 置为 null，布尔值转成字符串 `"true"`/`"false"`。作者在 `transform` 里统一把 Boolean 转成 String。
3. **每一行都做三组补值**，申请人和关联人都做：

| 补值后的列 | 优先级 |
|---|---|
| `birth_259D_1` | `birth_259D` → `birthdate_87D` |
| `gender_992L_1` | `gender_992L` → `sex_738L` |
| `relationshiptoclient_415T_1` | `relationshiptoclient_415T` → `relationshiptoclient_642T` |

4. **丢掉 11 列**（作者的 `cols_drop`）：

| 丢掉的列 | 原因 |
|---|---|
| `birthdate_87D`、`sex_738L`、`relationshiptoclient_642T` | 已经合并进上面的补值列（源码可证） |
| `contaddr_district_15M`、`registaddr_district_1083M`、`empladdr_district_926M` | 地区，取值很多（推断：容易过拟合） |
| `contaddr_zipcode_807M`、`registaddr_zipcode_184M`、`empladdr_zipcode_114M` | 邮编，原因同上（推断） |
| `personindex_1023L` | 只是表单上的填写顺序（推断：没有风险含义） |
| `isreference_387L` | 推断：信息基本已包含在其他角色字段里 |

### 4.2 申请人一行

取 `num_group1 = 0` 的行，保留剩下列中 `include = 1` 的 18 列，不做聚合。列名格式是 `<字段>_1_person1`，例如 `gender_992L_1_person1`。

| 类别 | 字段 |
|---|---|
| 个人 | `birth_259D`（已补值，天数）、`gender_992L`（已补值）、`education_927M`、`familystate_447L`、`language1_981M` |
| 就业和收入 | `empl_employedfrom_271D`（天数）、`empl_employedtotal_800L`、`empl_industry_691L`、`incometype_1044T`、`mainoccupationinc_384A` |
| 住房和地址 | `housetype_905L`、`contaddr_matchlist_1032L`、`contaddr_smempladdr_334L` |
| 角色和类型 | `persontype_1072L`、`role_1084L`、`role_993L`、`type_25L`、`safeguarantyflag_411L` |

**被删掉的 `include = 0` 字段：** `childnum_185L`、`housingtype_772L`、`maritalst_703L`、`persontype_792L`、`relationshiptoclient_415T`、`remitter_829L`。其中 `relationshiptoclient_415T` 是补值后的列，所以申请人的关系补值实际没有进入模型（见 1.2 的说明）。

**原因（推断）：** 每个案例只有一个申请人，这一行本身就是案例级的信息，不需要聚合。

### 4.3 关联人聚合

取 `num_group1 ≠ 0` 的行，按案例聚合。只聚合在 `features_2.csv` 里配置了聚合函数的字段，**`include = 0` 的字段也照样聚合**（源码可证：`include` 只影响申请人那一行）。

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `birth_259D`（已补值） | 生日 | min, max | 关联人里年龄最大和最小的 |
| `childnum_185L` | 子女数 | min, max | |
| `education_927M` | 教育程度 | mode, nuniquetotal | 关联人的主要教育程度，以及是否多样 |
| `gender_992L`（已补值） | 性别 | mode, nuniquetotal | |
| `housingtype_772L` | 住房类型 | mode, nuniquetotal | |
| `maritalst_703L` | 婚姻状况 | mode | |
| `persontype_1072L` | 人员类型 | mode | 关联人主要是哪一类 |
| `persontype_792L` | 人员类型（另一版本） | mode | |
| `relationshiptoclient_415T`（已补值） | 与申请人的关系 | mode | 最常见的关系，比如配偶或亲属 |
| `type_25L` | 联系方式类型 | mode | |

另外有一列 `case_id_1_count__person1`，记录关联人数，没有关联人为 0。

列名例子：`relationshiptoclient_415T_1_mode_person1` 是这个申请里最常见的关联人关系。

### 4.4 person2 聚合

`_process_person2` 先丢掉 5 列（源码可证）：`addres_district_368M`、`addres_role_871L`、`empls_employedfrom_796D`、`empls_employer_name_740M`、`relatedpersons_role_762T`。剩下 3 个字段，对案例下的全部记录聚合：

| 字段 | 含义 | 聚合 | 原因（推断） |
|---|---|---|---|
| `addres_zip_823M` | 地址邮编 | nunique, nuniquetotal | 申请里出现过几个不同的地址 |
| `conts_role_79M` | 联系人角色 | nunique, nuniquetotal | 联系人角色是否多样 |
| `empls_economicalst_849M` | 经济状态 | nunique, nuniquetotal | 就业状态是否多样 |

另外有一列 `case_id_1_count__person2`，记录条数。邮编在 person1 里被丢掉了，在这里却作为"有几个不同的值"保留下来：只统计个数，不用具体取值。

### 4.5 其他输出列和报表

| 列 | 说明 |
|---|---|
| `has_person1_applicant` | 是否有申请人行 |
| `has_person1_related` | 是否有关联人 |
| `has_person2` | 是否有 person2 记录 |

报表写在 `reports/04_person/`：

- `row_coverage.csv`：两个源的行数，以及有申请人、有关联人、有 person2 记录的案例数。
- `fallback_coverage.csv`：三组补值**分别按申请人和关联人统计**的补值前后缺失数。申请人行的 `relationshiptoclient_415T` 补值不进入模型（`include = 0`）。

**和前几个脚本的区别：** 为了缩短运行时间，04 先写表，再从写好的表里读回来做检查，特征只计算一遍。如果检查报错，表已经存在，需要先删表再重跑。

---

## 5. `train_credit_bureau_1_features`

这是最复杂的一个源，输出大约 710 列，列名后缀是 `_creditbureau1`。

### 5.1 数据的特殊结构

`credit_bureau_a_1` 的**每一行同时放着两份合同**：一份进行中（active）、一份已结清（closed），分别存在不同的列里。例如：

- 逾期天数：`dpdmax_139P` 是 active 合同的，`dpdmax_757P` 是 closed 合同的
- 合同结束日：`dateofcredend_289D` 是 active 合同的，`dateofcredend_353D` 是 closed 合同的

`credit_bureau_b_1` 是另一个征信来源，字段名不同。作者通过配置里的 `new_name` 把它的列改成 a1 的对应列名（共 29 个，例如 `contractdate_551D` → `dateofcredstart_181D`），然后当作同一个源处理（源码可证）。

### 5.2 用哪些行

作者是**按文件分别聚合**的，最后每个案例只保留第一个结果（`unique(keep="first")`，源码可证）。这带来两条规则，脚本在聚合前就把行筛好：

| 规则 | 影响 |
|---|---|
| 一个案例在 a1 里有记录，它在 b1 里的行就不用 | b1 只对没有 a1 记录的案例起作用 |
| a1 分成多个分片文件，一个案例如果跨了多个分片，只用第一个分片里的行 | 跨分片的案例会丢掉一部分合同 |

两条规则的影响可以从 `row_coverage.csv` 看出：`a1_rows_in_bronze` 和 `a1_rows_used`、`b1_rows_in_bronze` 和 `b1_rows_used` 的差（也包括没对上 `train_base` 的行）。

**原因（推断）：** 作者为了控制内存按文件分批处理，这两条规则是分批处理的副作用，不像是有意的特征设计。

### 5.3 行级处理

和前面一样：INNER JOIN 案例表；日期换成天数；`a55475b1` 置为 null。然后按 `_process_creditbureau1` 做以下处理（源码可证）：

**补值（2 个）**

| 派生列 | 优先级 | 原因（推断） |
|---|---|---|
| `totalamount_996A_2` | active 合同总额 → active 信用额度 `credlmt_935A` | 信用卡类合同没有"总额"，用额度代替 |
| `totalamount_6A_2` | closed 合同总额 → closed 信用额度 `credlmt_230A` | 同上 |

**年月拼成日期（4 个）：** 原始数据把"最大逾期发生的月份"分成年、月两列存。作者拼成当月 1 号，再换算成相对 `date_decision` 的天数，结果写进月份列的 `_1`，例如 `dpdmaxdatemonth_442T_1`。

**比率（12 个）**

| 派生列 | 公式 | 含义（推断） |
|---|---|---|
| `numberofoverdueinstlmax_1039L_2` / `_1151L_2` | 最大逾期期数 / 总期数 | 逾期期数占比 |
| `numberofoutstandinstls_59L_2` / `_520L_2` | 未还期数 / 总期数 | 剩余还款进度 |
| `dpdmax_139P_2` / `dpdmax_757P_2` | 最大逾期天数 / 逾期期数 | 平均每次逾期多久 |
| `debtoverdue_47A_2` | 逾期金额 / 未偿金额 | 债务中逾期的比例 |
| `outstandingamount_362A_2` | 未偿金额 / 合同总额（补值后） | 还剩多少没还 |
| `overdueamount_659A_2` / `overdueamount_31A_2` | 逾期金额 / 合同总额（补值后） | 逾期规模 |
| `dateofcredend_289D_3` / `_353D_3` | 结束日天数 / 合同数 | 含义不明确，照源码复现 |

`numberofoutstandinstls_520L_2` 用的是 closed 的分子除以 closed 的总期数（`229L`），`numberofoutstandinstls_59L_2` 用 active 的（`320L`），和源码一致。

**日期差（2 个）：** `dateofcredend_*_2` = 结束日 − 开始日，即合同期限。

**丢掉 13 列：** 8 个年/月原始列（已经拼成日期），以及 5 个只在 b1 里出现、后面没有用到的列（`credlmt_1052A`、`instlamount_892A`、`periodicityofpmts_997M/L`、`residualamount_3940956A`）。

### 5.4 拆成 active / closed 两种视图

这是作者的 `_handle_contracts_cb`（源码可证）。**每一行原始数据拆成两行**：

| 视图 | 包含的列 |
|---|---|
| active 行（`_active = true`） | 配置里标为 `active` 的列 + `common` 列 |
| closed 行（`_active = false`） | 配置里标为 `closed` 的列 + `common` 列 |

**同一个指标的 active 和 closed 两个版本合并成一列**，命名为 `<名称>_all_<序号>`。例如 `dpdmax_139P_1`（active）和 `dpdmax_757P_1`（closed）合并成 `dpdmax_all_1`。只有两个版本都存在时才合并，共 41 对。

只有一个版本的列（例如 `debtoutstand_525A_1` 只有 active）保留原名，在另一种视图里是 null。

**原因（推断）：** 合并以后，"逾期天数"这类指标可以跨所有合同统一聚合，不用分两套列各算一遍。

**排序：** 按 `dateofcredend_all_1`（合同结束日）升序，null 排最前，这决定 `last` 和 `rangelastfirst` 的结果。同一天的行，active 视图在前，再按 `num_group1` 排；作者对这种情况没有规定。

### 5.5 分组

| 分组 | 条件 | 聚合哪些列 | 列名后缀 |
|---|---|---|---|
| 全部 | 两种视图的所有行 | 全部 57 列 | 无 |
| active | `_active = true` | 仅数值列 | `_active` |
| closed | **作者的代码里也是 `_active = true`**（见下） | 仅数值列 | `_closed` |
| 近 5 年 | `dateofcredend_all_1` ≥ −1825 | 全部 57 列 | `_dateofcredend_all_5y` |

每组还有两列：

- `case_id_1_count_<后缀>`：行数。注意"全部"组每份原始数据算两行，因为它被拆成了两种视图。
- `source_1_<后缀>`：这个案例用的是 `credit_bureau_a_1` 还是 `credit_bureau_b_1`。

**最后，每个 `_active` 数值聚合列都除以对应的 `_closed` 列**，生成 `<X>_active_div_<X>_closed`，大约 130 列。

### 5.6 ⚠️ 作者代码里的 bug

源码中 closed 分组的过滤条件写成了 `df.filter(pl.col("active"))`，和 active 分组完全一样（源码可证，第 473 行）。后果是：

- 所有 `_closed` 列的值都和对应的 `_active` 列相同。
- 所有 `_div_` 列只会是 1 或者 null，不含任何信息。

脚本**默认按原样复现**（`CLOSED_FILTER_AS_AUTHOR = True`），这样输出和作者的模型输入一致。把它改成 `False` 就会按 closed 行过滤，可以用来评估修正后的效果。

### 5.7 聚合函数（在第 2.4 节基础上新增的）

| 函数 | 定义 | 说明 |
|---|---|---|
| `sum` | 求和 | **全为 null 时结果是 0**，不是 null。这和 polars 一致 |
| `last` | 按排序最后一个非空值 | 即结束日最晚的那份合同 |
| `rangelastfirst` | `last − first` | 从最早到最晚合同的变化 |
| `meannonzero` | 非零值的平均 | 例如只看实际有逾期的合同的平均逾期额 |

### 5.8 各字段的聚合

聚合方法取自 `features_2.csv`。合并后的 `_all` 列用 `_all` 自己那一行配置。规律如下：

| 类别 | 例子 | 典型聚合 | 原因（推断） |
|---|---|---|---|
| 逾期天数、逾期期数 | `dpdmax_all`、`numberofoverdueinstls_all` | min, max（部分加 mean, last） | max 是最严重的一次逾期 |
| 逾期金额 | `overdueamount_all`、`debtoverdue_47A` | mean, max；`debtoverdue` 另有 sum、meannonzero | 规模和最坏情况 |
| 合同金额、未偿金额 | `credlmt_all`、`debtoutstand_525A`、`totaloutstanddebtvalue_all` | min, max, mean，部分加 sum | sum 是总负债 |
| 利率、期数 | `nominalrate_all`、`numberofinstls_all` | min, max, mean, last | last 是最近一份合同的条件 |
| 日期 | `dateofcredstart_all`、`lastupdate_all` | min, max, mean | 征信历史有多长、离现在多近 |
| 分类 | `contractst_all`、`purposeofcred_all`、`subjectrole_all` | nunique, nuniquetotal, mode, last | 合同状态、用途、角色是否多样，以及最近一份是什么 |
| 金融机构 | `financialinstitution_all` | nunique, nuniquetotal | 在几家机构有过贷款 |

完整的字段与函数对应关系见脚本里的 `CONFIG`，共 158 项，和 `features_2.csv` 一一对应。

### 5.9 其他说明

- `has_creditbureau1`：这个案例是否有征信合同记录。
- 所有列名含 `case_id_` 的列，null 都补 0，包括计数列，以及计数列相除得到的 `_div_` 列（源码可证）。
- b1 的一个细节：b1 中"最大逾期金额发生的年份"映射到了 active 的年份列，月份却映射到了 closed 的月份列。所以对 b1 的行，这两个日期都拼不出来，结果为 null。这是作者配置造成的，脚本照样复现。

报表写在 `reports/05_credit_bureau_1/`：`row_coverage.csv` 记录各种行数和案例来源，`fallback_coverage.csv` 记录两组总额补值的效果。

---

## 6. `train_static_0_features`

`train_static_0` 是深度 0 的表，**每个案例本来就只有一行**，内容是本次申请的信息，以及贷款机构内部对这个客户的汇总统计（例如过去 24 个月的平均分期、最近一次审批的金额）。所以这个源**不需要聚合**，作者只做了清洗、补值和公式（`_process_static0`，源码可证）。

输出约 200 列，列名后缀是 `_static0`。

### 6.1 处理步骤

1. INNER JOIN `train_case_sources`，取 `date_decision` 和 person1 的申请人收入。
2. 清洗：日期换成天数，`a55475b1` 置为 null，`isbid*` 标志转成 0/1，其他布尔值转成字符串。
3. **丢掉 4 列**：`previouscontdistrict_112M`、`isdebitcard_729L`、`paytype1st_925L`、`paytype_783L`。它们在配置里也正好是 `include = 0`。
4. **两个补值**，见 6.2。
5. **36 个公式**，见 6.3。
6. LEFT JOIN 到全部案例，±inf 和 NaN 置为 null。

其余 163 个原始字段全部原样保留，列名格式是 `<字段>_1_static0`。

### 6.2 补值

| 派生列 | 优先级 | 是否进入模型 |
|---|---|---|
| `maininc_215A_2` | static0 收入 `maininc_215A` → person1 收入 `mainoccupationinc_384A` | ✓，同时是多个比率的分母 |
| `sumoutstandtotal_3546847A_2` | 未偿总额 → 未偿总额估计值 `sumoutstandtotalest_4493215A` | **✗**，被下一步的比率用同名列覆盖了（源码可证） |

第二个补值就是 01 里的 `outstanding_balance`。01 和这里的结论一致：它不进入模型，只在报表里保留覆盖率。

### 6.3 公式

**以当前贷款额 `credamount_770A` 为分母（推断：衡量相对本次贷款的规模）**

| 派生列 | 公式 |
|---|---|
| `annuity_780A_3` | 月供 / 贷款额 |
| `annuitynextmonth_57A_3` | 下月月供 / 贷款额 |
| `avglnamtstart24m_4525187A_2` | 近 24 个月平均放款额 / 贷款额 |
| `avgoutstandbalancel6m_4187114A_2` | 近 6 个月平均未偿余额 / 贷款额 |
| `disbursedcredamount_1113A_2` | 实际放款额 / 贷款额 |
| `downpmt_116A_2` | 首付 / 贷款额 |
| `maxoutstandbalancel12m_4187113A_2` | 近 12 个月最高未偿余额 / 贷款额 |
| `sumoutstandtotal_3546847A_3` | 未偿总额 / 贷款额 |
| `totinstallast1m_4525188A_2` | 上月分期总额 / 贷款额 |

**以收入为分母（推断：还款负担）**

| 派生列 | 公式 | 收入用哪个 |
|---|---|---|
| `annuity_780A_2` | 月供 / 收入 | 补值后的 `maininc_215A_2` |
| `credamount_770A_2` | 贷款额 / 收入 | 同上 |
| `currdebt_22A_2` | 当前债务 / 收入 | 同上 |
| `price_1097A_2` | 商品价格 / 收入 | 同上 |
| `sumoutstandtotal_3546847A_2` | 未偿总额 / 收入 | 同上 |
| `totinstallast1m_4525188A_3` | 上月分期总额 / 收入 | 同上 |
| `annuitynextmonth_57A_2` | 下月月供 / 收入 | person1 的收入 |
| `maininc_215A_3` | static0 收入 / person1 收入 | 两个收入来源是否一致 |

**本次月供和历史水平比较（推断：这次的负担比以往重还是轻）**

| 派生列 | 公式 |
|---|---|
| `annuity_780A_4` | 月供 / 近 24 个月平均分期 |
| `annuity_780A_5` | 月供 / 近 12 个月平均还款 |
| `annuity_780A_6`、`annuity_780A_7` | 月供 / 历史最高月供（两个版本） |
| `annuity_780A_8` | 月供 / 近 24 个月最高分期 |
| `annuity_780A_9` | 月供 / 近 3 个月最高还款 |
| `credamount_770A_3` | 贷款额 / 上次获批金额 |
| `credamount_770A_4` | 贷款额 / 近 6 个月最高放款额 |
| `currdebt_22A_3` | 当前债务 / 历史最高债务 |

**其他比率**

| 派生列 | 公式 | 说明 |
|---|---|---|
| `totalsettled_863A_2` | 已结清金额 / 总债务 | 还清了多少 |
| `inittransactionamount_650A_2` | 首笔交易额 / 月供 | |
| `interestrategrace_34L_2` | 宽限期利率 / 利率 | |
| `monthsannuity_845L_2` | 月数 / 月供 | 含义不明确，照源码复现 |
| `lastrepayingdate_696D_2` | 最后还款日 / 最后未付日（都是天数） | 两个日期偏移量相除，含义不明确，照源码复现 |

**差值（5 个）**

| 派生列 | 公式 | 含义（推断） |
|---|---|---|
| `datelastunpaid_3546854D_2` | 最后未付日 − 首次报价日 | |
| `dtlastpmtallstes_4499206D_2` | 最后还款日 − 最后未付日 | 欠款后多久还上 |
| `lastapprdate_640D_2` | 上次获批日 − 上次申请日 | 审批用了多久 |
| `lastrejectdate_50D_2` | 上次被拒日 − 上次申请日 | |
| `eir_270L_2` | 实际利率 − 名义利率 | 隐含的费用 |

### 6.4 报表

报表写在 `reports/06_static_0/`：

| 文件 | 内容 |
|---|---|
| `row_coverage.csv` | static0 行数，以及有/没有 static0 的案例数 |
| `fallback_coverage.csv` | 两个补值的效果，被覆盖的那个标记为 `used_by_model = false` |

---

## 7. `train_static_cb_0_features`

`train_static_cb_0` 也是深度 0，**每个案例一行，不需要聚合**。内容是外部征信机构和税务机构对客户的汇总。作者对它的处理很少（`_process_staticcb0`，源码可证）：

- 不丢任何列，52 个字段全部保留，列名格式是 `<字段>_1_staticcb0`
- 6 个补值
- 1 个比率

输出约 60 列，列名后缀是 `_staticcb0`。

### 7.1 字段内容

| 类别 | 字段 | 说明 |
|---|---|---|
| 个人信息 | `birthdate_574D`、`dateofbirth_337D`、`dateofbirth_342D`、`education_*`、`maritalst_*` | 同一信息的多个版本，用补值合并 |
| 征信查询次数 | `days30_165L`、`days90_310L`、`days120_123L`、`days180_256L`、`days360_512L`、`numberofqueries_373L` | 近 30 到 360 天被查询的次数（推断：短期内频繁查询往往说明客户在多处申请） |
| 被拒和取消次数 | `forweek_601L`、`formonth_118L`、`forquarter_462L`、`foryear_618L`、`for3years_128L`（被拒）；`forweek_1077L`、`formonth_206L`、`forquarter_1017L`、`foryear_818L`、`for3years_584L`（取消） | 按不同时间窗口统计，**外部机构已经汇总好了**，所以作者没有再做时间窗口 |
| 信用历史 | `fortoday_1092L`、`forweek_528L`、`formonth_535L`、`forquarter_634L`、`foryear_850L`、`for3years_504L`，以及四个季度的 `*quarter_*L` | |
| 税务 | `pmtaverage_*`（平均扣税额）、`pmtcount_*`、`pmtscount_423L`（扣税次数）、`pmtssum_45A`（扣税总额）、`assignmentdate_*`、`responsedate_*`、`requesttype_4525192L` | 扣税记录可以间接反映收入 |
| 风险评估 | `riskassesment_302T`、`riskassesment_940T` | 外部机构给出的违约概率或信用评估 |
| 其他 | `contractssum_5085716L`、`description_5085714M` | 合同总值、征信机构对客户的分类 |

### 7.2 补值

先置空 `a55475b1`、再补值，所以被掩码的主字段会落到下一个来源。每个补值都写入新的 `_2` 列，**原始输入列也保留**。

| 派生列 | 优先级 |
|---|---|
| `birthdate_574D_2` | `birthdate_574D` → `dateofbirth_337D` → `dateofbirth_342D` |
| `education_1103M_2` | `education_1103M` → `education_88M` |
| `maritalst_385M_2` | `maritalst_385M` → `maritalst_893M` |
| `pmtaverage_3A_2` | `pmtaverage_4527227A` → `pmtaverage_4955615A` → `pmtaverage_3A` |
| `pmtcount_4527229L_2` | `pmtcount_4527229L` → `pmtcount_4955617L` → `pmtcount_693L` |
| `assignmentdate_238D_2` | `assignmentdate_238D` → `assignmentdate_4527235D` → `assignmentdate_4955616D` |

这 6 个补值和 01 第 1.2 节里 `external_*` 开头的 6 条规则完全相同，01 已经在案例级别统计过它们的效果。`birthdate_574D_2` 还会在最后一步和 person1 的生日合并成 `birthdate_1_final0`。

**原因（推断）：** 带长数字编号的字段（如 `4527227A`、`4955615A`）看起来是数据源后来新增的版本，旧字段在新数据里逐渐变空。作者把新旧版本合成一列，让新旧数据保持连续。

注意 `pmtaverage` 的优先级是**新版本在前**（`4527227A` → `4955615A` → `3A`），而输出列名用的是旧字段名 `pmtaverage_3A_2`。

### 7.3 比率

| 派生列 | 公式 | 含义（推断） |
|---|---|---|
| `pmtssum_45A_2` | 扣税总额 / person1 申请人收入 | 税务记录和申报收入是否一致 |

源码在这一行上方有一行被注释掉的 `_scale_amount` 调用，说明作者试过别的缩放方式，最后用了直接相除。

### 7.4 报表

报表写在 `reports/07_static_cb_0/`：

| 文件 | 内容 |
|---|---|
| `row_coverage.csv` | 行数，以及有/没有 staticcb0 的案例数 |
| `fallback_coverage.csv` | 6 个补值的补值前后缺失数。这里按 staticcb0 的行统计，01 按全部案例统计，所以两边的 `rows` 不同 |

---

## 8. `train_credit_bureau_2_features`

征信的**每月还款记录**：每份合同（`num_group1`）下有多个月份（`num_group2`）。结构和第 5 节的 creditbureau1 相同，同样是一行里并排放着 active 和 closed 两份合同。列名后缀是 `_creditbureau2`，约 200 列。

### 8.1 和 creditbureau1 相同的部分

- a2 和 b2 当作同一个源处理。b2 有 2 列改成 a2 的列名。
- 按文件聚合带来的规则：在 a2 有记录的案例不用 b2；跨多个 a2 分片的案例只用第一个分片（见 5.2）。
- 拆成 active 和 closed 两种视图，成对的列合并成 `_all`，共 9 对。
- 分组有 active、closed、全部，加上每个 active 聚合除以对应 closed 聚合的 `_div_` 列。
- **closed 分组的 bug 一样存在**：过滤条件也是 `active`（见 5.6），默认照原样复现。

注意 b2 的改名：`pmts_dpdvalue_108P` 改成了 `pmts_overdue_1140A`（逾期金额），`pmts_pmtsoverdue_635A` 改成了 `pmts_dpd_1073P`（逾期天数）。按字段后缀看（Kaggle 约定 `P` 表示逾期天数、`A` 表示金额），这两个映射像是**对调了**。不过字段说明本身比较含糊，无法确定。这是作者配置里的 `new_name`，脚本照样复现。

### 8.2 行级处理（`_process_creditbureau2`，源码可证）

| 派生列 | 公式 | 含义（推断） |
|---|---|---|
| `pmts_month_706T_1`、`pmts_month_158T_1` | 年、月拼成当月 1 号，再换算成相对 `date_decision` 的天数 | 还款发生在多久以前 |
| `pmts_dpd_1073P_2`、`pmts_dpd_303P_2` | 逾期天数 > 7 记 1，否则记 0 | 是否逾期超过一周 |
| `pmts_dpd_1073P_3`、`pmts_dpd_303P_3` | 逾期天数 > 15 记 1，否则记 0 | 是否逾期超过半个月 |

逾期天数缺失时，标志记为 0，不是 null（polars `when/otherwise` 的行为）。4 个年、月原始列随后丢掉。

### 8.3 分组

| 分组 | 条件 | 列名后缀 |
|---|---|---|
| 全部 | 所有行 | 无 |
| active / closed | 同第 5 节，仅数值列 | `_active`、`_closed` |
| 近 1 年 | `pmts_month_all_1` ≥ −360 | `_pmts_month_all_1y` |
| 近 1 个月 | `pmts_month_all_1` ≥ −30 | `_pmts_month_all_1m` |

时间窗口用 360 天而不是 365 天，是源码里写的 `30*12`。

**原因（推断）：** 还款记录是按月的，最近的还款行为比几年前的更能反映当前风险，所以作者用了比其他源（5 年）短得多的窗口。

### 8.4 聚合

| 字段 | 含义 | 聚合 |
|---|---|---|
| `pmts_dpd_all`（含 `_2`、`_3` 标志） | 每月逾期天数 | max, mean, sharenonzero, min |
| `pmts_overdue_all` | 每月逾期金额 | max, mean, sum, std, min |
| `pmts_month_all` | 还款月份 | min, max, mean |
| `collater_valueofguarantee_all` | 抵押物价值 | max, mean, min |
| `collater_typofvalofguarant_all`、`collaterals_typeofguarante_all`、`subjectroles_name_all` | 抵押类型、角色 | nunique, nuniquetotal, mode, last |

新增的两个函数：

- `sharenonzero` = 非零值个数 / 非空值个数。用在逾期天数上，就是"有逾期的月份占比"；用在 `_2`、`_3` 标志上，就是"逾期超过 7 天或 15 天的月份占比"。
- `std` = 样本标准差，反映逾期金额的波动。

**排序：** `last` 按 `pmts_month_all_1` 排序，即最近一个月的值。这张表是所有源里最大的（a2 有上亿行，拆成两种视图后再翻倍），所以脚本**不用窗口函数预先排序**，而是在聚合时用 `max_by(值, struct(排序键))` 直接取最后一个，省掉一次全表排序。

---

## 9. `train_debitcard_deposit_other_features`

三个很小的深度 1 源，**各自独立聚合**，结果放在一张表里。列名后缀分别是 `_debitcard1`、`_deposit1`、`_other1`。

| 源 | 字段 | 聚合 | 时间窗口 |
|---|---|---|---|
| debitcard1 | 近 180 天平均余额、近 180 天流水、近 30 天流水 | min, max, sum | 按开卡日期，近 5 年 |
| | 开卡日期 `openingdate_857D` | min, max, mean, count | |
| deposit1 | 存款金额 `amount_416A` | min, max, sum | 按开户日期，近 5 年 |
| | 到期日 `contractenddate_991D` | min, max | |
| | 开户日期 `openingdate_313D` | min, max, count | |
| other1 | 借记卡转入/转出、存款余额、存款转入 | min, max, sum | 无 |
| | 存款转出 `amtdepositoutgoing_4809442A` | min, max, sum, mean | |

other1 有两个公式（源码可证）：

| 派生列 | 公式 | 含义（推断） |
|---|---|---|
| `amtdebitincoming_4809443A_2` | 借记卡转入 − 转出 | 借记卡账户的净流入 |
| `amtdepositincoming_4809444A_2` | 存款转入 − 转出 | 存款账户的净流入 |

**原因（推断）：** 这三个源反映客户在本机构的存款和流水，也就是还款能力。sum 是总规模，min/max 是波动范围，count 是账户数。

这些聚合都不依赖行的顺序。`sum` 在全为 null 时结果是 0，和 polars 一致。

---

## 10. `train_tax_registry_features`

三个税务登记源，分别记录雇主替客户代扣的税款。**各自独立聚合**，放在一张表里，列名后缀分别是 `_taxregistrya1`、`_taxregistryb1`、`_taxregistryc1`。

| 源 | 金额 | 日期 | 丢掉的列（源码可证） |
|---|---|---|---|
| a | `amount_4527230A`：min, max, mean, last | `recorddate_4527225D`：max, min, mean | 雇主名称 `name_4527232M` |
| b | `amount_4917619A`：min, max, mean, last | `deductiondate_4917603D`：max, min, last | 雇主名称 `name_4917606M` |
| c | `pmtamount_36A`：min, max, mean, last | `processingdate_168D`：max, min, last, mean | 雇主名称 `employername_160M` |

**原因（推断）：** 代扣税款间接反映收入水平和就业稳定性。`last` 是最近一笔扣税额；雇主名称取值太多，所以丢掉。

### 10.1 分组

| 分组 | 条件 | 列名后缀 |
|---|---|---|
| 近 1 个月 | 日期 ≥ −30 天 | `_1month` |
| 全部 | 所有行 | 无 |
| 近 5 年（只有 a 源） | `recorddate_4527225D` ≥ −1825 天 | `_recorddate_4527225D_5y` |

近 5 年这一组来自作者通用的日期窗口循环：`features_2.csv` 里 `date_col = 1` 的列都会加一个 5 年窗口，税务源里只有 a 源的 `recorddate_4527225D` 被标了（源码可证）。旧版脚本漏了这一组，现在已补上。

`last` 按各自的日期列排序（作者对三个源都设了 `col_sort`，源码可证）。

### 10.2 ⚠️ 作者代码里的 bug

c 源的近 1 个月过滤条件写成了 `pmtamount_36A_1 >= -30`，**用的是金额而不是日期**（源码可证）。金额基本都是正数，所以这个条件几乎保留了所有行，c 源的 `_1month` 特征实际上约等于全部历史。

脚本默认照原样复现（`TAX_C_FILTER_AS_AUTHOR = True`）。改成 `False` 就会按 `processingdate_168D` 过滤。`row_coverage.csv` 里的 `taxregistryc1_rows_in_1month_group` 可以看出影响有多大：在 bug 下，它会接近总行数。

---

## 11. `gold_volkova.train_features`（最终宽表，gold 层）

脚本是 `notebooks/gold/volkova/01_train_features.py`。它从 `silver_volkova` 读取各源表，输出到 `gold_volkova` schema，存储路径在 `gold/volkova` 下。schema 需要事先建好；volume `gold_volkova.files` 不存在时，脚本会自动创建。

把 02 到 10 的所有特征表合并成一张建模用的表，再做作者 `FinalDataProcessor` 里的跨源处理（源码可证）。

### 11.1 合并

1. 以 `train_case_sources` 的 `case_id`、`date_decision`、`WEEK_NUM`、`target` 为主表。作者最后会丢掉 `MONTH`，所以这里不取。
2. 依次 LEFT JOIN 9 张特征表。
3. **审计列不进入最终表**：所有 `has_*` 标记留在各自的源表里，需要时再 join 回来。
4. 任何重名列都会让脚本报错停止，避免悄悄覆盖。

### 11.2 跨税务源特征（8 列）

对 a、b、c 三个税务源的同名统计量做合并，全部历史和近 1 个月各一套：

| 派生列 | 计算方式 | 说明 |
|---|---|---|
| `amount_1_mean_taxregistry1` | (a 均值 + b 均值 + c 均值) / 3 | **缺失的源按 0 计，分母固定是 3**（polars `sum_horizontal` 跳过 null） |
| `amount_1_min_taxregistry1` | 三者中的最小值 | 跳过缺失 |
| `amount_1_max_taxregistry1` | 三者中的最大值 | 跳过缺失 |
| `amount_1_last_taxregistry1` | 三个 `last` 中的**最大值** | 源码就是用 `max_horizontal`，不是"最近的一笔" |

`_1month` 版本同理，例如 `amount_1_mean_1month_taxregistry1`。

**注意：**

- 只有一个税务源的客户，mean 会被除以 3，偏小。这是作者的写法。
- 三个源都没有数据时，mean 是 0 而不是 null。这是按 polars `sum_horizontal` 的行为推断的，**没有实际验证**。

### 11.3 最终补值（3 列）

| 派生列 | 优先级 | 和 01 的关系 |
|---|---|---|
| `birthdate_1_final0` | staticcb0 生日（已补值） → person1 生日（已补值） | 同 01 的 `applicant_birthdate` |
| `employed_length_1_final0` | 最近一次历史申请的就业开始日 → person1 就业开始日 | 同 02 |
| `mainoccupationinc_1_final0` | static0 收入 → person1 收入 | 同 01 的 `applicant_main_income` |

### 11.4 最后处理

- 列名含 `case_id_` 的列 null 补 0，±inf 和 NaN 置为 null。各源表已经做过，这里再做一遍保证这一步自身完整。
- 作者还会把字符串转成 categorical，并按 `WEEK_NUM`、`case_id` 排序。Delta 表里字符串保持字符串，也不保存行顺序，**建模时需要自己处理这两点**。

### 11.5 报表

报表写在 gold volume 的 `reports/01_train_features/`：

| 文件 | 内容 |
|---|---|
| `column_inventory.csv` | 每一列来自哪张源表，派生列标为 `gold_01_train_features` |
| `columns_per_source.csv` | 每张源表贡献了多少列 |
| `final_fallbacks.csv` | 3 个最终补值各补回了多少案例 |

---

## 附：作者代码中发现的问题汇总

| 位置 | 问题 | 影响 | 脚本中的开关 |
|---|---|---|---|
| 01 / 06 `sumoutstandtotal_3546847A_2` | 补值结果被同名比率覆盖 | 补值不进入模型 | 无，只在报表中标出 |
| 01 / 04 `relationshiptoclient_415T` | 补值后因 `include = 0` 从申请人行删除 | 申请人关系补值不进入模型 | 无，只在报表中标出 |
| 02 `dtlastpmt_581D_2` | 补值结果被同名日期差覆盖 | 补值不进入模型 | 无，只在报表中标出 |
| 05 / 08 closed 分组 | 过滤条件写成 `active` | `_closed` 列重复 `_active`，`_div_` 列无信息 | `CLOSED_FILTER_AS_AUTHOR` |
| 02 / 05 / 08 按文件聚合 | b 源只用于 a 源没有的案例；跨分片案例只用第一个分片 | 丢失部分记录 | 无，报表统计使用的行数 |
| 08 b2 改名 | 按后缀看，逾期天数和逾期金额的映射像是对调了（未确认） | b2 数据可能进入错误的列 | 无 |
| 10 c 源近 1 个月 | 按金额而不是日期过滤 | `_1month` 几乎等于全部历史 | `TAX_C_FILTER_AS_AUTHOR` |
| gold 04 `reduce_mem_usage` | 数值列范围允许时转成 float16（约 3 位有效数字） | 进入模型的金额、天数被四舍五入，例如 -3651 天变成 -3652，12345.67 变成 12344 | 无，照原样复现 |
| gold 01 税务均值 | 缺失源按 0 计，分母固定为 3 | 只有部分源的客户均值偏小 | 无 |
