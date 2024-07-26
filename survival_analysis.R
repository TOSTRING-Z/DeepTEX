library(survival)
library(survminer)
library(tibble)
library(dplyr)
library(tibble)

args <- commandArgs(trailingOnly = TRUE)
name <- args[1]

# TEX risk score
risk_scores <- read.csv(paste0("output/bulk/model/risk_scores-", name, ".csv"))

# Clinical information
clinical <- read.csv("data/clinical.csv")
clinical <- clinical[risk_scores$X,]

risk_scores <- risk_scores$X0
# Divide into high-risk group and low-risk group based on risk score
threshold <- median(risk_scores)
# threshold <- 0.5
print(threshold)
clinical$group = if_else(risk_scores > threshold,1,0)
clinical$status = if_else(clinical$status == 0,1,2)
# Kaplan Meier survival analysis
fit <- survfit(Surv(clinical$time,clinical$status) ~ clinical$group)

pdf(file = paste0("output/bulk/model/Kaplan-Meier-",name,".pdf"),onefile=F)
ggsurvplot(
  fit,
  data = clinical,
  conf.int = TRUE,
  pval = TRUE,
  surv.median.line = "hv",
  palette = "hue",
  risk.table = TRUE,
  legend.labs = c("Low", "High"),
  risk.table.height = 0.25,
  ggtheme = theme_classic2()
)
dev.off()

# Nomogram
library(rms)

clinical <- cbind(clinical,risk_scores)
head(clinical)

ddist <- datadist(clinical)
options(datadist = 'ddist')

res.cox <- cph(Surv(time,status) ~ stage + gender + age + risk_scores, data=clinical, surv=T, x=T, y=T, time.inc=365 * 3)
surv <- Survival(res.cox)

nom <- nomogram(res.cox,
  fun = list(function(x) surv(365, x), function(x) surv(365 * 3, x), function(x) surv(365 * 10, x)),
  lp = F,
  funlabel = c("1-year survival", "2-year survival", "3-year survival")
)

pdf(file = paste0("output/bulk/model/Nomogram-",name,".pdf"))
plot(nom)
dev.off()