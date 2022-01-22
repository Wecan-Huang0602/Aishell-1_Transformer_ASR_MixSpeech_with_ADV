import sys
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from prepare import prepare_aishell
import pandas as pd
import json
import os

logger = logging.getLogger(__name__)


# 定義字典
class WordDict:
    def __init__(self, hparams):
        self.word2index = {
            '<PAD>': hparams["pad_index"],
            '<BOS>': hparams["bos_index"],
            '<EOS>': hparams["eos_index"],
            '<UNK>': hparams["unk_index"]
        }
        self.word2count = {
            '<PAD>': 0,
            '<BOS>': 0,
            '<EOS>': 0,
            '<UNK>': 0
        }
        self.index2word = {
            hparams["pad_index"]: '<PAD>',
            hparams["bos_index"]: '<BOS>',
            hparams["eos_index"]: '<EOS>',
            hparams["unk_index"]: '<UNK>'
        }
        self.n_words = len(self.index2word)
        train_data_df = pd.read_csv(hparams["train_data"])

        for line in train_data_df['transcript'].to_numpy():
            self.addSentence(line)

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 定義 Tokenizer
class Tokenizer:
    def __init__(self, wd):
        self.wd = wd

    def encode(self, x):
        result = []
        for word in x:
            # 如果字沒有在字典裡就加入 <UNK>
            if word not in self.wd['word2index']:
                result.append(self.wd['word2index']['<UNK>'])
                continue
            result.append(self.wd['word2index'][word])
        return result

    def decode(self, x):
        result = []
        for idx in x:
            result.append(self.wd['index2word'][str(idx)])
        return result


# Define training procedure
class ASR(sb.core.Brain):
    def get_batch_data(self, batch):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Add augmentation if specified
        if hasattr(self.hparams, "SpeedPerturb"):
            wavs_speed_90 = self.hparams.SpeedPerturb['speed_90'](wavs)
            wavs_speed_100 = wavs
            wavs_speed_110 = self.hparams.SpeedPerturb['speed_110'](wavs)

            wav_lens = torch.cat([
                wav_lens * (wavs_speed_90.size(1) / wavs_speed_110.size(1)),
                wav_lens * (wavs_speed_100.size(1) / wavs_speed_110.size(1)),
                wav_lens
            ])
            wavs = pad_sequence(
                [
                    *wavs_speed_90,
                    *wavs_speed_100,
                    *wavs_speed_110
                ],
                batch_first=True
            )
            tokens_bos = torch.cat(
                [
                    tokens_bos,
                    tokens_bos,
                    tokens_bos
                ],
                dim=0
            )
            tokens_eos = torch.cat([tokens_eos, tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [
                    tokens_eos_lens,
                    tokens_eos_lens,
                    tokens_eos_lens
                ],
                dim=0
            )
            tokens = torch.cat([tokens, tokens, tokens], dim=0)
            tokens_lens = torch.cat(
                [
                    tokens_lens,
                    tokens_lens,
                    tokens_lens
                ],
                dim=0
            )
        if hasattr(self.modules, "env_corrupt"):
            wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])
            tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        if hasattr(self.hparams, "SpecAugment"):
            feats = self.hparams.SpecAugment(feats)

        batch_data = {
            'feats': feats,
            'tokens_bos': tokens_bos,
            'tokens_eos': tokens_eos,
            'tokens': tokens,
            'wav_lens': wav_lens,
            'tokens_eos_lens': tokens_eos_lens,
            'tokens_lens': tokens_lens
        }
        return batch_data

    def forward_asr_model(self, feats, wav_lens, tokens_bos_current, tokens_bos_previous=None):
        src = self.modules.CNN(feats)
        if self.mix_stage:
            enc_out, pred_current, pred_previous = self.modules.Transformer.forward_mix(
                src=src,
                tgt_current=tokens_bos_current,
                tgt_previous=tokens_bos_previous,
                wav_len=wav_lens,
                pad_idx=self.hparams.pad_index
            )
        else:
            enc_out, pred_current = self.modules.Transformer(
                src=src,
                tgt=tokens_bos_current, 
                wav_len=wav_lens,
                pad_idx=self.hparams.pad_index
            )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        ctc_prob = self.hparams.softmax(logits)
        ctc_logprob = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        # Current
        pred_current = self.modules.seq_lin(pred_current)
        seq_prob_current = self.hparams.softmax(pred_current)
        seq_logprob_current= self.hparams.log_softmax(pred_current)

        if self.mix_stage:
            # Previous
            pred_previous = self.modules.seq_lin(pred_previous)
            seq_prob_previous = self.hparams.softmax(pred_previous)
            seq_logprob_previous = self.hparams.log_softmax(pred_previous)

            return (
                ctc_prob, 
                ctc_logprob, 
                seq_prob_current, 
                seq_logprob_current, 
                seq_prob_previous, 
                seq_logprob_previous
            )
        else:
            return ctc_prob, ctc_logprob, seq_prob_current, seq_logprob_current

    def compute_asr_loss(
        self,
        ctc_logprob,
        seq_logprob,
        tokens,
        tokens_eos,
        tokens_lens,
        tokens_eos_lens,
        wav_lens
    ):
        loss_ctc = self.hparams.ctc_cost(
            log_probs=ctc_logprob,
            targets=tokens,
            input_lens=wav_lens,
            target_lens=tokens_lens
        )
        loss_seq = self.hparams.seq_cost(
            log_probabilities=seq_logprob,
            targets=tokens_eos,
            length=tokens_eos_lens
        )
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        return loss

    def run_mixspeech(
        self,
        feats,
        tokens,
        tokens_bos,
        tokens_eos,
        tokens_lens,
        tokens_eos_lens,
        wav_lens
    ):
        current_batch_size = feats.size(0)

        if self.mix_stage:
            # print('Mix')
            
            # ***Mix Current &  Previous Batch Data***
            wav_lens = torch.max(
                wav_lens*feats.size(1),
                self.previous_batch_data['wav_lens'] * self.previous_batch_data['feats'].size(1)
            )
            wav_lens = wav_lens / torch.max(wav_lens)

            feats = pad_sequence(
                [
                    *feats,
                    *self.previous_batch_data['feats']
                ],
                batch_first=True
            )
            feats = (
                (1-self.mix_previous_weight) * feats[:current_batch_size]
                + self.mix_previous_weight * feats[current_batch_size:]
            )

            # ***Forward ASR Model***
            (
                ctc_prob, 
                ctc_logprob, 
                seq_prob_current, 
                seq_logprob_current, 
                seq_prob_previous, 
                seq_logprob_previous

            ) = self.forward_asr_model(
                feats=feats,
                wav_lens=wav_lens,
                tokens_bos_current=tokens_bos,
                tokens_bos_previous=self.previous_batch_data['tokens_bos']
            )

            # ***Compute Losses***
            # Compute Current ASR Loss
            current_asr_loss = self.compute_asr_loss(
                ctc_logprob=ctc_logprob,
                seq_logprob=seq_logprob_current,
                tokens=tokens,
                tokens_eos=tokens_eos,
                tokens_lens=tokens_lens,
                tokens_eos_lens=tokens_eos_lens,
                wav_lens=wav_lens
            )
            # Compute Previous ASR Loss
            previous_asr_loss = self.compute_asr_loss(
                ctc_logprob=ctc_logprob,
                seq_logprob=seq_logprob_previous,
                tokens=self.previous_batch_data['tokens'],
                tokens_eos=self.previous_batch_data['tokens_eos'],
                tokens_lens=self.previous_batch_data['tokens_lens'],
                tokens_eos_lens=self.previous_batch_data['tokens_eos_lens'],
                wav_lens=wav_lens
            )

            # Compute Current-Previous KL-Loss 
            ctc_logprob_pad = pad_sequence(
                [
                    *ctc_logprob,
                    *self.previous_batch_data['ctc_logprob']
                ],
                batch_first=True
            )
            ctc_prob_pad = pad_sequence(
                [
                    *ctc_prob,
                    *self.previous_batch_data['ctc_prob']
                ],
                batch_first=True
            )

            ctc_kl_loss = (
                self.hparams.kl_div(ctc_logprob_pad[:current_batch_size], ctc_prob_pad[current_batch_size:])
                + self.hparams.kl_div(ctc_logprob_pad[current_batch_size:], ctc_prob_pad[:current_batch_size])
            )
            seq_kl_loss = (
                self.hparams.kl_div(seq_logprob_previous, self.previous_batch_data['seq_prob'])
                + self.hparams.kl_div(self.previous_batch_data['seq_logprob'], seq_prob_previous)
            )
            kl_loss = (
                self.hparams.ctc_weight * ctc_kl_loss
                + (1-self.hparams.ctc_weight) * seq_kl_loss
            )

            # Combine all Loss 
            loss = (
                (1-self.mix_previous_weight) * current_asr_loss
                + self.mix_previous_weight * (previous_asr_loss + (self.hparams.adv_kl_weight*kl_loss))
            )
            (loss / self.hparams.gradient_accumulation).backward()
        else:
            # print('Clean')

            # 只跑 Clean
            # init feats adv noise
            feats_delta = torch.zeros_like(feats)
            feats_delta.requires_grad_()
            feats = feats + feats_delta

            # ***Forward ASR Model***
            ctc_prob, ctc_logprob, seq_prob_current, seq_logprob_current = self.forward_asr_model(
                feats=feats,
                wav_lens=wav_lens,
                tokens_bos_current=tokens_bos
            )

            # ***Compute ASR Loss***
            loss = self.compute_asr_loss(
                ctc_logprob=ctc_logprob,
                seq_logprob=seq_logprob_current,
                tokens=tokens,
                tokens_eos=tokens_eos,
                tokens_lens=tokens_lens,
                tokens_eos_lens=tokens_eos_lens,
                wav_lens=wav_lens
            )
            (loss / self.hparams.gradient_accumulation).backward(retain_graph=True)

            # ***Update Noise***
            # update and clip for wav delta
            feats_delta_grad = feats_delta.grad.clone().detach().float()
            denorm = torch.norm(feats_delta_grad.view(feats_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            feats_delta_step = (feats_delta_grad / denorm).to(feats_delta)

            # Rescale Feats Delta Step to 0.01~0.001
            feats_delta_step = self.auto_scale(
                x=feats_delta_step,
                min_v=self.hparams.adv_spec_min,
                max_v=self.hparams.adv_spec_max
            )
            feats_delta = (feats_delta + feats_delta_step).detach()

            # add new adv noise to feats
            feats = (feats + feats_delta).detach()

        return (
            loss,
            feats,
            wav_lens,
            ctc_prob, 
            ctc_logprob, 
            seq_prob_current, 
            seq_logprob_current
        )

    def scale_value(self, x, min_v, max_v):
        x_max = x.max()
        x_min = x.min()

        x = (((x-x_min) / (x_max-x_min)) * (max_v-min_v)) + min_v

        return x

    def auto_scale(self, x, min_v, max_v):
        x = x

        for i in range(x.size(0)):
            x_temp = x[i]

            x_temp[x_temp > 0] = self.scale_value(
                x=x_temp[x_temp > 0],
                min_v=min_v,
                max_v=max_v
            )
            x_temp[x_temp < 0] = self.scale_value(
                x=x_temp[x_temp < 0],
                min_v=-max_v,
                max_v=-min_v
            )

        return x

    def run_train_stage(self, batch):
        batch_data = self.get_batch_data(batch=batch)
        current_batch_size = batch_data['feats'].size(0)

        # 決定這個 Batch 的 mix_previous_weight
        self.mix_previous_weight = torch.rand(1).item() * self.hparams.max_mix_previous_weight

        '''
        如果當前的 Batch Size 跟上一個 Batch 的 Batch Size 一樣，
        並且上個 Stage 不是 Mix-Stage，才可以做 Mixspeech & ADV
        '''
        if (current_batch_size == self.previous_batch_data['batch_size']) and (not self.mix_stage):
            self.mix_stage = True
        else:
            self.mix_stage = False

        # Run ASR and Noise Stage
        (
            loss,
            feats,
            wav_lens,
            ctc_prob, 
            ctc_logprob, 
            seq_prob, 
            seq_logprob

        ) = self.run_mixspeech(
            feats=batch_data['feats'],
            tokens=batch_data['tokens'],
            tokens_bos=batch_data['tokens_bos'],
            tokens_eos=batch_data['tokens_eos'],
            tokens_lens=batch_data['tokens_lens'],
            tokens_eos_lens=batch_data['tokens_eos_lens'],
            wav_lens=batch_data['wav_lens']
        )

        # Update current batch to previous batch
        self.previous_batch_data = batch_data
        self.previous_batch_data['batch_size'] = current_batch_size
        self.previous_batch_data['feats'] = feats
        self.previous_batch_data['wav_lens'] = wav_lens
        self.previous_batch_data['ctc_prob'] = ctc_prob
        self.previous_batch_data['ctc_logprob'] = ctc_logprob
        self.previous_batch_data['seq_prob'] = seq_prob
        self.previous_batch_data['seq_logprob'] = seq_logprob

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            if self.check_gradients(loss):
                self.optimizer.step()

            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        loss = self.run_train_stage(batch=batch)

        return loss.detach()

    def compute_forward(self, feats, tokens_bos, wav_lens, stage):
        """Just for VALID/TEST stage forward computations."""
        if stage != sb.Stage.TRAIN:
            # Forward ASR Model
            src = self.modules.CNN(feats)
            enc_out, pred = self.modules.Transformer(
                src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
            )

            # output layer for ctc log-probabilities
            logits = self.modules.ctc_lin(enc_out)
            ctc_logprob = self.hparams.log_softmax(logits)

            # output layer for seq2seq log-probabilities
            pred = self.modules.seq_lin(pred)
            seq_logprob = self.hparams.log_softmax(pred)

            # Compute outputs
            hyps = None
            if stage == sb.Stage.VALID:
                current_epoch = self.hparams.epoch_counter.current
                if current_epoch % self.hparams.valid_search_interval == 0:
                    # for the sake of efficiency, we only perform beamsearch with limited capacity
                    # and no LM to give user some idea of how the AM is doing
                    hyps, _ = self.hparams.valid_search(
                        enc_out.detach(), wav_lens)
            elif stage == sb.Stage.TEST:
                hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return ctc_logprob, seq_logprob, hyps

    def compute_objectives(
        self,
        ctc_logprob,
        seq_logprob,
        tokens,
        tokens_eos,
        tokens_lens,
        tokens_eos_lens,
        wav_lens,
        stage,
        wrd,
        hyps,
        ids
    ):
        """Just for VALID/TEST stage loss computations."""
        if stage != sb.Stage.TRAIN:
            loss = self.compute_asr_loss(
                ctc_logprob=ctc_logprob,
                seq_logprob=seq_logprob,
                tokens=tokens,
                tokens_eos=tokens_eos,
                tokens_lens=tokens_lens,
                tokens_eos_lens=tokens_eos_lens,
                wav_lens=wav_lens
            )

            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                predicted_words_list = []
                target_words_list = [[c for c in seq] for seq in wrd]

                for prediction in hyps:
                    # Decode token terms to words
                    predicted_tokens = self.tokenizer.decode(
                        prediction
                    )
                    predicted_words_list.append(predicted_tokens)

                self.cer_metric.append(
                    ids=ids,
                    predict=predicted_words_list,
                    target=target_words_list,
                )

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(
                log_probabilities=seq_logprob,
                targets=tokens_eos,
                length=tokens_eos_lens,
            )
        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        batch = batch.to(self.device)
        ids = batch.id
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        wrd = batch.wrd

        with torch.no_grad():
            # compute features
            feats = self.hparams.compute_features(wavs)
            current_epoch = self.hparams.epoch_counter.current
            feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

            # forward modules
            ctc_logprob, seq_logprob, hyps = self.compute_forward(
                feats=feats,
                tokens_bos=tokens_bos,
                wav_lens=wav_lens,
                stage=stage
            )
            loss = self.compute_objectives(
                ctc_logprob=ctc_logprob,
                seq_logprob=seq_logprob,
                tokens=tokens,
                tokens_eos=tokens_eos,
                tokens_lens=tokens_lens,
                tokens_eos_lens=tokens_eos_lens,
                wav_lens=wav_lens,
                stage=stage,
                wrd=wrd,
                hyps=hyps,
                ids=ids
            )

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages acccording current stage
            steps = self.hparams.noam_annealing.n_steps
            lr = self.hparams.noam_annealing.current_lr

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "epoch": epoch},
                min_keys=["loss"],
                num_to_keep=self.hparams.keep_nbest_models,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evalation stage
            # delete the rest of the intermediate checkpoints
            # CER is set to 0 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"loss": 0, "epoch": epoch},
                min_keys=["loss"],
                num_to_keep=1,
            )

    def on_fit_start(self):
        super().on_fit_start()

        self.previous_batch_data = {
            'batch_size': 0
        }

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
            max_num_checkpoints=self.hparams.keep_nbest_models
        )

        # Log Top-K CKPTS Infor
        self.hparams.train_logger.log_stats(
            stats_meta={
                '\nFind {} ckpts'.format(len(ckpts)): ''.join(
                    ['\nEpoch: {}, Loss: {}'.format(
                        c.meta['epoch'], c.meta['loss']) for c in ckpts]
                )
            }
        )

        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={
            "data_folder": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"], replacements={
            "data_folder": data_folder},
    )
    # valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={
            "data_folder": data_folder},
    )
    # test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data


if __name__ == '__main__':
    # ==顯示有哪些GPU==
    print('-'*30)
    print('All GPU:')
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    print('-'*30)

    # ==讀入配置檔案==
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    print(run_opts)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # ==生成實驗的輸出目錄==
    # # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # ==載入預訓練的參數==
    # # We download and pretrain the tokenizer
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])
    # print('Done')

    # ==顯示模型參數量==
    print("Total parameter of the model = " + str(sum(p.numel()
          for p in hparams['model'].parameters())))

    # ==資料前處理==
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # ==建立 Tokenizer 和 Lexicon ==
    # 建立並保存 lexicon
    wd = WordDict(hparams=hparams)
    print('字典大小: ', len(wd.word2index))

    # 把字典保存起來
    lexicon_data = {'word2index': wd.word2index, 'index2word': wd.index2word}
    with open(hparams["lexicon_file"], 'w', encoding='utf-8') as f:
        json.dump(lexicon_data, f)

    # 讀入 lexicon 檔
    with open(hparams["lexicon_file"], "r") as f:
        wd = json.load(f)

    lexicon_size = len(wd['word2index'])
    print('Load 字典大小: ', lexicon_size)

    # 定義 Tokenizer
    tokenizer = Tokenizer(wd=wd)
    # a = tokenizer.encode('早安你好')
    # a, tokenizer.decode(a)

    # ==定義 Dataset 的東西==
    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # ==初始化 speechbrain 的核心東西 Brain Class==
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # ==訓練模型==
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    print('Done')

    # ==測試模型==
    # 關掉 Data Parallel
    # # Close the data parallel
    # run_opts['data_parallel_backend'] = False
    # print(run_opts)

    # # Trainer initialization
    # asr_brain = ASR(
    #     modules=hparams["modules"],
    #     hparams=hparams,
    #     run_opts=run_opts,
    #     checkpointer=hparams["checkpointer"],
    # )

    # print('Data parallel: ', asr_brain.data_parallel_backend)

    # # adding objects to trainer:
    # asr_brain.tokenizer = tokenizer

    # Testing
    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_folder"], "cer_test.txt"
    )

    asr_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"]
    )

    # Val
    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_folder"], "cer_dev.txt"
    )

    asr_brain.evaluate(
        valid_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"]
    )
