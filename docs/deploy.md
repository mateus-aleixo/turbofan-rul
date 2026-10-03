# Deploying to AWS

From zero (no AWS account) to a live endpoint. One-time setup is ~30 minutes;
after that every deploy is `git tag && git push --tags`. The AWS resources keep
the project's original name, `conformal-rul`.

## What it costs

| Service | Free tier | This project |
|---|---|---|
| Lambda | 1M requests + 400k GB-s / month, forever | ~0 at demo traffic |
| API Gateway (HTTP) | 1M requests / month, first 12 months | ~0; $1/M after year 1 |
| ECR | 500 MB storage, first 12 months | image ≈ 0.4 GB → ~$0.04/month after year 1 |
| CloudWatch logs | 5 GB ingest / month | ~0 |

The API stage is throttled (5 req/s, burst 10) and a $5/month budget alarm
emails at 80 %: worst case is capped twice over.

## 1. Create the account

1. <https://aws.amazon.com> → Create account (email, credit card required).
2. Sign in as **root** once: enable **MFA on the root user** (IAM → Security
   credentials). Never create root access keys.
3. Create the working identity: IAM → Users → `admin`, enable console access,
   attach `AdministratorAccess`, enable MFA. Create an **access key** (CLI use)
   and store it nowhere but `aws configure`.

## 2. Install tooling

```powershell
winget install Amazon.AWSCLI Hashicorp.Terraform
aws configure   # access key, secret, region: eu-west-1, output: json
```

## 3. Bootstrap the infrastructure

The Lambda references the container image, so the registry must exist and hold
one image before the first full apply:

```powershell
cd infra
terraform init
terraform apply -target=aws_ecr_repository.api -var budget_email=YOU@example.com

# push the first image
$acct = aws sts get-caller-identity --query Account --output text
aws ecr get-login-password | docker login --username AWS --password-stdin "$acct.dkr.ecr.eu-west-1.amazonaws.com"
docker build -t "$acct.dkr.ecr.eu-west-1.amazonaws.com/conformal-rul:latest" ..
docker push "$acct.dkr.ecr.eu-west-1.amazonaws.com/conformal-rul:latest"

terraform apply -var budget_email=YOU@example.com
```

`terraform output` prints the four values that matter: `api_endpoint`,
`ecr_repository_url`, `lambda_function_name`, `deploy_role_arn`.

## 4. Wire up GitHub deploys (OIDC, no stored keys)

```powershell
gh variable set AWS_ROLE_ARN --body (terraform output -raw deploy_role_arn)
gh variable set AWS_REGION   --body eu-west-1
gh variable set API_ENDPOINT --body (terraform output -raw api_endpoint)
```

## 5. Deploy and verify

```powershell
git tag v0.1.0 && git push --tags   # or: gh workflow run deploy
```

```bash
curl "$API/health"
curl "$API/models"
curl -X POST "$API/predict" -H "Content-Type: application/json" -d @- <<'EOF'
{"subset": "FD001", "coverage": 90,
 "cycles": [{"setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
   "s_01": 518.67, "s_02": 641.82, "s_03": 1589.7, "s_04": 1400.6, "s_05": 14.62,
   "s_06": 21.61, "s_07": 554.36, "s_08": 2388.06, "s_09": 9046.19, "s_10": 1.3,
   "s_11": 47.47, "s_12": 521.66, "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
   "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0, "s_20": 39.06,
   "s_21": 23.419}]}
EOF
```

(One cycle is enough: the service left-pads and says so via `"padded": true`.
The first request after idle pays a ~2–4 s cold start; that is Lambda, not the
model.)

## 6. Teardown

```powershell
terraform destroy -var budget_email=YOU@example.com
```

## Troubleshooting

- **`sts:AssumeRoleWithWebIdentity` denied in Actions**: two causes seen in
  practice: (a) the workflow ran from a branch or fork the trust policy doesn't
  cover (it trusts `main` and `v*` tags only); (b) **the sub-claim format**:
  GitHub pins numeric ids into the token
  (`repo:owner@id/name@id:ref:...`), so a trust policy written against the
  classic id-less format matches nothing. The Terraform here builds the
  id-pinned form from `github_owner_id`/`github_repo_id`; if you fork this,
  set those to your own ids (`gh api users/<you> --jq .id`,
  `gh api repos/<you>/<repo> --jq .id`). Debug by dumping the claims:
  request the token in a step and decode its payload with `jq '{sub, aud}'`.
- **`lambda wait function-updated` fails with AccessDenied**: the waiter
  polls `lambda:GetFunctionConfiguration`; a minimal role with only
  `UpdateFunctionCode`/`GetFunction` breaks exactly there. Already granted
  here.
- **Image push denied**: ECR login expired (12 h); rerun the login command.
- **502 from the API**: almost always the container failed to start; check
  CloudWatch → log group `/aws/lambda/conformal-rul`.
- **Why infra isn't applied by CI**: the deploy role deliberately can't touch
  IAM/API Gateway/budgets. Infrastructure changes are a local, reviewed
  `terraform apply`; CI only ships application code. Widening that role is a
  choice, not an accident.
