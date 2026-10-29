# Azure Databricks 部署事项清单

目标：把 lakehouse 管道（raw → bronze → silver → gold）跑在 **Azure Databricks** 上，
数据放在我们自己的 ADLS Gen2，gold 层导出给 Colab。README 里的架构不用改。

预算：大约 $100–250（按 2 个月、集群累计约 50 小时估算）。价格以
[Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/) 和 Databricks 官网为准。

> 第 0 步可选：先注册 [Databricks Free Edition](https://www.databricks.com/learn/free-edition)，
> 试一下能否跑通 bronze 层的 ingestion。如果够用，下面的步骤可以全部跳过，费用为 $0。

---

## 1. Azure 账号与安全（第 1 天）

- [ ] 登录 [Azure Portal](https://portal.azure.com)，确认有一个可以付费的 subscription（Pay-As-You-Go）
  - Azure for Students 也能用，但 vCPU 配额比较低，可能开不了多台 worker，需要申请提高配额
- [ ] 账号开启 **MFA**
- [ ] 区域统一选 **East US**，所有资源都放进同一个 resource group：`rg-home-credit`
- [ ] **设置 Budget 告警**（Cost Management → Budgets → Add）
  - 月预算 $100，实际花费达到 50%、80%、100% 时发邮件
  - 另外开一个 *Forecasted* 100% 告警（预测会超支时提前通知）
- [ ] （可选）本地安装 Azure CLI：`brew install azure-cli && az login`

## 2. 创建 Databricks workspace（第 1–2 天）

- [ ] Portal → Create a resource → **Azure Databricks**
  - Resource group：`rg-home-credit`
  - Workspace name：`dbw-home-credit-audit`
  - Region：East US
  - Pricing tier：**Premium**（Unity Catalog 需要）
  - Networking 保持默认（不需要 VNet injection）
- [ ] 建好后点 *Launch Workspace*，记下 URL（格式是 `https://adb-<数字>.<数字>.azuredatabricks.net`）
- [ ] 确认 Unity Catalog 已经自动启用（Catalog Explorer 里能看到 workspace 默认的 catalog）
- [ ] 建项目 catalog 和 schema：
  ```sql
  CREATE CATALOG IF NOT EXISTS home_credit;
  CREATE SCHEMA IF NOT EXISTS home_credit.bronze;
  CREATE SCHEMA IF NOT EXISTS home_credit.silver;
  CREATE SCHEMA IF NOT EXISTS home_credit.gold;
  CREATE VOLUME IF NOT EXISTS home_credit.bronze.raw_files;  -- 存放 Kaggle 原始文件
  ```
  - 如果 `CREATE CATALOG` 报错说需要 managed location，就先做第 7 步的存储配置，再加上 `MANAGED LOCATION 'abfss://...'`

> 注意：Azure 会自动创建一个 **managed resource group**（名称以 `databricks-rg-` 开头），里面放 VNet 和存储账号，
> 有时还会有 NAT Gateway。建好后进去看一眼有哪些计费资源。**不要手动删除里面的东西**，清理方法见第 8 步。

## 3. 控制计算成本（第 2 天）

- [ ] 建一个 **cluster policy**，所有成员都必须使用：
  - `autotermination_minutes` 固定为 20
  - worker 用 **Spot**（`azure_attributes.availability = SPOT_WITH_FALLBACK_AZURE`），driver 用 on-demand
  - 实例类型限定为 `Standard_E4ds_v5` / `Standard_E8ds_v5`（内存优化型），worker 最多 4 台
- [ ] 优先使用 **Serverless** compute（不用管 VM，按用量计费）；如果没有 Serverless，再用一个小的 **all-purpose** 集群调试；bronze/silver/gold 的批处理用 **Jobs** 集群跑（DBU 单价便宜很多）
- [ ] 每周看一次 Azure Cost Management 和 Databricks Account Console → Usage

## 4. 导入 Kaggle 数据（第 2–3 天）

- [ ] 每个成员在 Kaggle 竞赛页面点 **Accept rules**（否则无法下载）
- [ ] 把 Kaggle API token 存进 Databricks secret，**不要**写进 notebook：
  ```bash
  databricks secrets create-scope kaggle
  databricks secrets put-secret kaggle username
  databricks secrets put-secret kaggle key
  ```
- [ ] 在 `notebooks/bronze/` 下写一个下载 notebook，把数据下载到 Volume：
  ```python
  import os
  os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get("kaggle", "username")
  os.environ["KAGGLE_KEY"] = dbutils.secrets.get("kaggle", "key")
  %pip install kaggle
  !kaggle competitions download -c home-credit-credit-risk-model-stability -p /Volumes/home_credit/bronze/raw_files
  !cd /Volumes/home_credit/bronze/raw_files && unzip -o -q *.zip
  ```
- [ ] 确认原始 parquet/csv 文件都已经在 Volume 里，然后开始写 bronze ingestion

## 5. 本地 VS Code 连接 + 团队协作（第 3 天）

- [ ] VS Code Databricks 扩展：侧边栏 Databricks 图标 → Create configuration → 填 workspace URL → 用 **OAuth (user to machine)** 登录
  - 扩展会在 repo 根目录生成 `databricks.yml`，可以提交；`.databricks/` 目录已经被 `.gitignore` 排除
- [ ] 添加队友：workspace → Settings → Identity and access → Users
  - 队友必须在同一个 Entra ID 租户里（比如都用 NYU 账号），否则要先在 Entra ID 里邀请为 guest
- [ ] 建一个 group（比如 `home-credit-team`），授予权限：
  ```sql
  GRANT USE CATALOG, USE SCHEMA, SELECT, MODIFY, CREATE TABLE
    ON CATALOG home_credit TO `home-credit-team`;
  ```
- [ ] 每个人在 User Settings → Linked accounts 绑定 GitHub，然后通过 Repos/Git folders clone 本 repo（流程见 README）
- [ ] 把 cluster policy 的使用权限授予这个 group，同时**收回**自由创建集群的权限

## 6. 管道开发（第 3 天起）

- [ ] `notebooks/bronze/`：原始文件 → Delta 表（`home_credit.bronze.*`）
- [ ] `notebooks/silver/`：清洗、去重、关联各张历史表
- [ ] `notebooks/gold/`：特征聚合 → `home_credit.gold.*`
- [ ] 用 Databricks Workflows 把三层串成一个 Job，跑在 Jobs 集群上

## 7. 把 gold 层导出到 Colab

- [ ] 在 `rg-home-credit` 里建一个 **Storage account**（勾选 *Hierarchical namespace* = ADLS Gen2），比如 `sthomecreditgold`，再建一个 container `gold-export`，关闭匿名访问
- [ ] 建一个 **Access Connector for Azure Databricks**，在存储账号上给它分配 *Storage Blob Data Contributor* 角色
- [ ] 在 Unity Catalog 里用这个 Access Connector 建 storage credential 和 external location，然后在 `notebooks/colab_export/` 里把 gold 表写成 Parquet：
  ```python
  spark.table("home_credit.gold.features").write.mode("overwrite") \
      .parquet("abfss://gold-export@sthomecreditgold.dfs.core.windows.net/gold/features/")
  ```
- [ ] 给 container 生成一个**只读的 SAS token**（权限只勾 Read + List，设置过期时间）
- [ ] Colab 里用 `google.colab.userdata`（Secrets）保存 SAS token，然后读取：
  ```python
  # pip install adlfs
  import pandas as pd
  from google.colab import userdata
  df = pd.read_parquet(
      "abfs://gold-export/gold/features/",
      storage_options={"account_name": "sthomecreditgold", "sas_token": userdata.get("AZURE_SAS")},
  )
  ```

## 8. 项目结束后清理（必须做）

- [ ] 把最终的 gold 数据和模型结果下载到本地或 Google Drive 备份
- [ ] 删除整个 resource group `rg-home-credit`
  - 会一起删掉 workspace、存储账号和 Access Connector；`databricks-rg-*` 这个 managed resource group 也会随 workspace 自动删除
- [ ] 回到 Portal → Resource groups，确认 `rg-home-credit` 和 `databricks-rg-*` 都已经消失
- [ ] 作废还没过期的 SAS token（删除存储账号后会自动失效）
- [ ] 下个月账单出来后核对一遍，确认没有残留费用
