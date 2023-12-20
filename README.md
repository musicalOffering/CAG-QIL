# CAG-QIL: Context-Aware Actionness Grouping via Q Imitation Learning for Online Temporal Action Localization
[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html)

# OAD pretrain step
1. Prepare your video features (tensor shape: [L, D], where L: length of the video, D: feature dimension) and labels (tensor shape: [L, N+1], where N: number of class.) in `OAD/data/feature`, `OAD/data/label` folder respectively. </br> See `make_label_thumos14.py` for detailed format of the label. </br>
We provide THUMOS14 features and label [here](https://drive.google.com/file/d/1HtOf6xL9h1jMdUWa9M3m8i_6LHyvpmjH/view?usp=drive_link).

2. You have to train <u>two separate</u> OAD model, binary (or class-agnostic) and class-aware model respectively. 
Run `oad_train.py` with `model_name=class` for the class-aware model, and `model_name=binary` for class-agnostic model. Other argument will raise error message. We provide pretrained OAD models [here](https://drive.google.com/file/d/13Q-yzrwkMWCu2uOlq0tJPMtnhnhojiuj/view?usp=drive_link).

3. With trained OAD models, we transform the original feature sequence into probability sequence, which will be used as a CAG-QIL agent's input. To do this, you have to execute 4 command: </br> 

    ```
    python rl_envmaker.py --load_model=epoch_10_0.5834.pt --subset=train --class_agno=False 
    python rl_envmaker.py --load_model=epoch_10_0.5834.pt --subset=test --class_agno=False 
    python rl_envmaker.py --load_model=epoch_10_0.4286.pt --subset=train --class_agno=True 
    python rl_envmaker.py --load_model=epoch_10_0.4286.pt --subset=test --class_agno=True 
    ```

    Concatenation of class-agnostic and class-aware sequence is done by executing following:

    ```
    python unify.py --binary_path=rlenv_train_binary --class_path=rlenv_train_class --save_path=rlenv_train

    python unify.py --binary_path=rlenv_test_binary --class_path=rlenv_test_class --save_path=rlenv_test

    ```

4. (Optional) You can assess baseline performance with following commands: </br>

    ```
    python baseline_maker.py
    python json_maker.py
    ``` 
    </br>
    This will generate `baseline.json` in the `proposals` folder. See next section for its evaluation.



# CAG-QIL agent train step
- Move `rlenv_train` and `rlenv_test` to the CAG_QIL folder. Also, copy `rlenv_train` and `OAD/data/label` to the `CAG_QIL/factory`.
## Making Expert database
- Run `make_expert_database.py` and `separate.py` sequentially. This will result in `expert_ones_24_24.pkl` and `expert_zeros_24_24.pkl` in `factory` folder.
## Train CAG-QIL agent
- Run `dqn_main.py`
## Choose the best model and measure performance
1. We conduct validation process.
You may choose several candidate models from saved models by observing the running f1 score.
Run following commands for each model.
    ```
    ##example
    python run.py --mode=eval --load_model=1100000step_actor.pt
    python json_maker.py --mode=eval --file_name=eval2.json     
    python hungarian_metric.py --mode=eval --pred_file=eval2.json 
    ```
    It will yield a validation f1 score for each model. 

2. With the model with highest validation f1 score, conduct following:
    ```
    ##example
    python run.py --mode=test --load_model=1300000step_actor.pt 
    python json_maker.py --mode=test --file_name=test.json     
    python hungarian_metric.py --mode=test --pred_file=test.json 
    ```
    You can also measure the baseline performance by simply moving the `baseline.json` to the proposals folder and run the last command.

