cd home/monika/dyishere/project/MyResearch/NavDP/baselines/bridgedp
python bridgedp_server.py --port 8888 \
    --checkpoint /home/monika/dyishere/project/MyResearch/InternNav/checkpoints/bridgedp_train/ckpts/checkpoint-15927bridgedp.ckpt \
    --sigma_base 0.0817 --sigma_goal 0.1 --n_prior_tokens 4 \
    --exec_num_waypoints 24 --exec_waypoint_spacing 0.15
