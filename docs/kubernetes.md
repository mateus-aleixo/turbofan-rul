# Kubernetes

The same image that runs on AWS Lambda, deployed as a Kubernetes Deployment, with
the properties that a Deployment is supposed to give you asserted in CI rather
than assumed.

Nothing in `src/` changed to make this work. The container already spoke plain
HTTP because the Lambda Web Adapter requires it, so the port that Lambda pokes at
is the port kubelet probes.

## What is here

```
deploy/
  helm/conformal-rul/     the chart
  kind/cluster.yaml       single node, NodePort 30080 mapped to the host
  kind/values.yaml        the three things a CI cluster cannot honour
  scripts/no-downtime.sh  the disruption tests
.github/workflows/k8s.yml lint, render, validate, then run all of it on kind
```

## The decisions worth defending

**`maxUnavailable: 0`, `maxSurge: 1`.** Capacity never drops below the declared
replica count during a rollout. It costs one pod of headroom while rolling and
buys an update nobody notices.

**A `preStop` sleep of 5 seconds.** This is the one that is easy to leave out and
impossible to debug afterwards. Endpoint removal and pod termination are
concurrent, not sequential: kube-proxy learns a pod is Terminating at roughly the
same moment the pod gets SIGTERM, so a pod that exits immediately can still be
handed requests by a node whose iptables rules have not caught up. Sleeping in
`preStop` keeps the process serving while that propagates.

**Three probes, not one.** `startupProbe` covers ONNX session load with a
generous budget, so `livenessProbe` can stay tight for the rest of the pod's life
instead of being permanently lenient to accommodate a slow boot.
`readinessProbe` gates traffic. Collapsing these into one probe means either slow
starts get killed or a wedged process never does.

**`readOnlyRootFilesystem: true`, non-root, all capabilities dropped.** The
service reads a model registry and writes nothing, so the filesystem is mounted
read-only and `/tmp` is an `emptyDir`. If a future change needs to write
somewhere, the failure is immediate and local rather than a surprise in
production.

**`automountServiceAccountToken: false`.** The service never calls the Kubernetes
API. The ServiceAccount exists only so a cloud identity (IRSA on EKS, Workload
Identity on GKE) can be bound to it later without touching the Deployment.

**HPA on CPU.** An inference request here is CPU work start to finish. There is
no queue to measure and no GPU to saturate, so CPU utilisation against the
request is the honest signal. Scale up with no stabilisation window, scale down
after 120 seconds: latency is what a caller feels, and a spare pod for two
minutes is cheaper than a cold start per burst.

**A PDB that only covers voluntary disruption.** `minAvailable: 1` protects
against node drains and cluster upgrades. It does nothing during a rolling
update (that is `maxUnavailable`) and nothing during a crash (that is the
scheduler). Saying so here is cheaper than explaining it in an incident.

## What CI actually proves

`helm lint`, then both value sets rendered and validated with `kubeconform`, then
a kind cluster where the image is side-loaded and the chart installed. After that
the two claims above are tested rather than asserted:

- **`no-downtime.sh rollout`** starts a poller against the NodePort, restarts
  every pod, and fails the build if a single probe returns anything but 200. It
  also fails if the pod set did not change, so a no-op rollout cannot pass by
  doing nothing, and if fewer than 20 probes covered the window, so a green
  result cannot come from an idle poller.
- **`no-downtime.sh kill-pod`** deletes a pod under the same load.

The poller deliberately uses a NodePort mapped to the host rather than
`kubectl port-forward`, because port-forward binds to one pod and dies with it,
which would report exactly the downtime the test exists to disprove.

Local run, 2026-09-04, kind v0.25.0, Kubernetes v1.31, two replicas:

```
mode: rollout    requests: 169   failures: 0
mode: kill-pod   requests:  39   failures: 0
helm test rul --logs
  health:     {'status': 'ok', 'version': '0.1.0'}
  predict OK: 125.0 {'coverage_nominal': 90, 'coverage_measured': 0.89, ...}
```

`helm test` runs inside the cluster using the service image itself rather than
pulling `curl` or `busybox`, so the test needs no registry access at all.

## Running it

```bash
make k8s-up        # kind cluster, build, load, install
make k8s-verify    # both disruption tests, then helm test
make k8s-down      # delete the cluster
```

## Lambda or Kubernetes

Both are wired up, and for this service Lambda is still the better answer. The
traffic is bursty and low volume, the image is small and torch-free, cold starts
land around 320 ms warm, and the whole thing costs cents per month with no
control plane to run or upgrade. A Kubernetes cluster for one FastAPI service
is a worse deal.

Kubernetes wins when the assumptions change: sustained traffic where per-request
pricing stops being cheap; GPU inference or long-lived model state, neither of
which fits a function; several services that need one network, one identity model
and one deployment story; a hard requirement to run on premises or in a specific
region; or a per-request runtime above what a function will give you.

The point of having both is being able to say which one a workload needs, and why,
with the chart to back it up rather than an opinion.
