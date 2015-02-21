require "rake/clean"

CLEAN

#
# Workaround function for using /bin/bash instead of /bin/sh, as suggested by
# http://stackoverflow.com/a/17052592
#
# Example:
#   bash %q( cat < <( seq 1 10 ) )
#
def bash(cmd)
  IO.popen(["/bin/bash", "-c", cmd]) { |io| io.read }
end

task :default=> [] do
  sh './evaluations.py heldout -m ./mdl/polarity2.0.mdl -c ./data/sfu_review_corpus/sfu_review_corpus.dat'
end

#
# Train the polarity2.0 model (and vocab).
#
file "mdl/polarity2.0.mdl" => ["data/polarity2.0/polarity2.0.csv"] do |t|
  src = t.prerequisites[0]
  vocab_target = t.name.pathmap("%X.vocab.csv")
  sh "./make_model.py -d #{src} -m #{t.name} -V #{vocab_target}"
end


#
# Make the sfu_review_corpus training csv data
#
file "data/sfu_review_corpus/sfu_review_corpus.csv" => \
  "data/sfu_review_corpus/SFU_Review_Corpus_Raw/MOVIES" do |t|
  src = t.prerequisites[0]

  # concatenate each review file and assign binary labels.
  sh "ls -1 #{src}/no*.txt | ./make_train_data.py -e cp1252 -l 0 > #{t.name}"
  sh "ls -1 #{src}/yes*.txt | ./make_train_data.py -e cp1252 -l 1 >> #{t.name}"
end


#
# Make the polarity2.0 training csv data
#
file "data/polarity2.0/polarity2.0.csv" => ["data/polarity2.0/txt_sentoken/"] do |t|
  src = t.prerequisites

  # concatenate each review file and assign binary labels.
  sh "ls -1 #{src[0]}/neg/* | ./make_train_data.py -l 0 > #{t.name}"
  sh "ls -1 #{src[0]}/pos/* | ./make_train_data.py -l 1 >> #{t.name}"
end

CLEAN << ["data/polarity2.0/polarity2.0.csv",
          "data/sfu_review_corpus.csv",
          "mdl/polarity2.0.mdl"]
