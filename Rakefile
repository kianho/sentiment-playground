require "rake/clean"

CLEAN

task :default => ["mdl/polarity2.0.mdl"] do
end

#
# Train the polarity2.0 model (and vocab).
#
file "mdl/polarity2.0.mdl" => ["data/polarity2.0/polarity2.0.dat"] do |t|
  src = t.prerequisites[0]
  vocab_target = t.name.pathmap("%X.vocab.dat")
  sh "./make_model.py -d #{src} -m #{t.name} -V #{vocab_target}"
end


#
# Make the sfu_review_corpus normalised training data
#
file "data/sfu_review_corpus/sfu_review_corpus.dat" => \
  "data/sfu_review_corpus/SFU_Review_Corpus_Raw/MOVIES" do |t|
  src = t.prerequisites[0]

  # assign negative and positive labels.
  sh "ls -1 #{src}/no*.txt | ./make_train_data.py -e cp1252 -l 0 > #{t.name}"
  sh "ls -1 #{src}/yes*.txt | ./make_train_data.py -e cp1252 -l 1 >> #{t.name}"
end


#
# Make the polarity2.0 training data
#
file "data/polarity2.0/polarity2.0.dat" => \
  ["data/polarity2.0/txt_sentoken/neg", "data/polarity2.0/txt_sentoken/pos"] do |t|
  
  target = t.name
  prereqs = t.prerequisites

  `ls -1 #{prereqs[0]}/* | ./make_train_data.py -l 0 > #{target}`
  `ls -1 #{prereqs[1]}/* | ./make_train_data.py -l 1 >> #{target}`
end

CLEAN << ["data/polarity2.0/polarity2.0.dat",
          "data/sfu_review_corpus.dat",
          "mdl/polarity2.0.mdl"]

#
# Snippets for later
#
# file "data/blah.txt" => ["data/poldata.README.2.0"] do |t|
#   cp t.prerequisites[0], t.name
# end
#
# t.prerequisites.each do |p|
#   sh "ls -1 #{p}/* | sort -n | xargs -I{} cat {} >> #{t.name}"
# end
# `ls -1 #{t.prerequisites[0]}/* | sort -n | xargs -I{} cat {} | awk '{print "0 " $0}' > #{t.name}`
# `ls -1 #{t.prerequisites[1]}/* | sort -n | xargs -I{} cat {} | awk '{print "1 " $0}' >> #{t.name}`
