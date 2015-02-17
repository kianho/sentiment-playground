require "rake/clean"

CLEAN

task :default => ["mdl/polarity2.0.mdl"] do
end

#
# Train the polarity2.0 model.
#
file "mdl/polarity2.0.mdl" => ["data/polarity2.0/polarity2.0.norm.dat"] do |t|
  target = t.name
  src = t.prerequisites[0]

  sh "./make_model.py -d #{src} -o #{t.name}"
end


#
# Make the sfu_review_corpus normalised training data
#
file "data/sfu_review_corpus/sfu_review_corpus.norm.dat" => \
  "data/sfu_review_corpus/SFU_Review_Corpus_Raw/MOVIES" do |t|
  src = t.prerequisites[0]

  # assign negative and positive labels.
  sh "ls -1 #{src}/no*.txt | ./make_train_data_sfu_review_corpus.py -l 0 > #{t.name}"
  sh "ls -1 #{src}/yes*.txt | ./make_train_data_sfu_review_corpus.py -l 1 | tail -n+2 >> #{t.name}"
end


#
# Make the polarity2.0 normalised training data
#
# Normalise and aggregate the polarity 2.0 dataset such that each line contains
# the class label and a single review.
#
file "data/polarity2.0/polarity2.0.norm.dat" => \
  "data/polarity2.0/polarity2.0.raw.dat" do |t|

  sh "./make_train_data_polarity2.0.py < #{t.prerequisites.first} > #{t.name}"
end

#
# Make the polarity2.0 raw training data
#
file "data/polarity2.0/polarity2.0.raw.dat" => \
  ["data/polarity2.0/txt_sentoken/neg", "data/polarity2.0/txt_sentoken/pos"] do |t|
  
  # Prepend the class label, cross-validation, and originating html doc-id to each line.
  # - neg. label -> '0', pos. label -> '0').
  # - the cross-validation and html doc-id are taken from the basename of each
  #   .txt file.
  #
  # Each line adopts the format:
  #
  #   <label><SEP><cv>_<docid><SEP><...raw sentence tokens separated by spaces>
  #
  # For example:
  #
  #   0@@SEP@@cv000_29416@@SEP@@plot : two teen couples go to a church party , drink and
  #   0@@SEP@@cv000_29416@@SEP@@they get into an accident . 
  #   0@@SEP@@cv000_29416@@SEP@@one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares . 
  #   0@@SEP@@cv000_29416@@SEP@@what's the deal ? 
  #   0@@SEP@@cv000_29416@@SEP@@watch the movie and " sorta " find out . . . 
  #
  # where "@@SEP@@" is the output column separator (see code below).
  target = t.name
  prereqs = t.prerequisites
  sep = "@@SEP@@"

  `echo "label#{sep}id#{sep}sentence" > #{target}`
  `ls -1 #{prereqs[0]}/* | parallel --gnu -k "sed 's/^/0#{sep}{/.}#{sep}/g' {}" >> #{target}`
  `ls -1 #{prereqs[1]}/* | parallel --gnu -k "sed 's/^/1#{sep}{/.}#{sep}/g' {}" >> #{target}`
end

CLEAN << ["data/polarity2.0/polarity2.0.raw.dat", "data/sfu_review_corpus.norm.dat"]

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
