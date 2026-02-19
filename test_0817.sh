for i in $(seq 0 2); do

python train_simple.py --extra_config ./config/adapt_exp/WIKI_TALK/TGN_base${i}.yml --logfile results/wiki_talk_tgn_base${i}_0817.log | tee results/output_wiki_talk_tgn_base${i}_0817.log;

python train_simple.py --extra_config ./config/adapt_exp/WIKI_TALK/TGN_adaptive${i}.yml --logfile results/wiki_talk_tgn_adaptive${i}_0817.log | tee results/output_wiki_talk_tgn_adaptive${i}_0817.log;

python train_simple.py --extra_config ./config/adapt_exp/WIKI_TALK/APAN_base${i}.yml --logfile results/wiki_talk_apan_base${i}_0817.log | tee results/output_wiki_talk_apan_base${i}_0817.log;

python train_simple.py --extra_config ./config/adapt_exp/WIKI_TALK/APAN_adaptive${i}.yml --logfile results/wiki_talk_apan_adaptive${i}_0817.log | tee results/output_wiki_talk_apan_adaptive${i}_0817.log;

done
