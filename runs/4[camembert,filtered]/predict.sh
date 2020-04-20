#!/bin/bash
#SBATCH --job-name=pred
#SBATCH --output=pred.out
#SBATCH --error=pred.err
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=1
#SBATCH --mem 90GO
python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.full.conll \
	--model model \
	--device cuda:0 \
	--export pred.full.full.conll \
	2> pred.err > pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.full.conll -s pred.full.full.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.full.conll -s pred.full.full.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out

python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.connected.conll \
	--model model \
	--device cuda:0 \
	--export pred.full.connected.conll \
	2>> pred.err >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.connected.conll -s pred.full.connected.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/full.connected.conll -s pred.full.connected.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out

python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.full.conll \
	--model model \
	--device cuda:0 \
	--export pred.ftb.full.conll \
	2>> pred.err >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.full.conll -s pred.ftb.full.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.full.conll -s pred.ftb.full.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out

python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.connected.conll \
	--model model \
	--device cuda:0 \
	--export pred.ftb.connected.conll \
	2>> pred.err >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.connected.conll -s pred.ftb.connected.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/ftb.connected.conll -s pred.ftb.connected.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out

python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.full.conll \
	--model model \
	--device cuda:0 \
	--export pred.sqeuoia.full.conll \
	2>> pred.err >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.full.conll -s pred.sequoia.full.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.full.conll -s pred.sequoia.full.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out

python /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/predict.py \
	--data /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.connected.conll \
	--model model \
	--device cuda:0 \
	--export pred.sequoia.connected.conll \
	2>> pred.err >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.connected.conll -s pred.sequoia.connected.conll.simple -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
TreebankAnalytics eval -g /mnt/beegfs/home/corro/repos/deep-syntactic-dependency-parsing-wip/data/sequoia/test/sequoia.connected.conll -s pred.sequoia.connected.conll.structured_no_node -c /mnt/beegfs/home/corro/repos/TreebankAnalytics/config.yaml >> pred.out
