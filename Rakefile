require "rake/clean"

CLEAN

task :default => ["data/txt_sentoken.norm.dat"] do
end

#
# Normalise and aggregate the polarity 2.0 dataset such that each line contains
# the class label and a single review.
#
file "data/txt_sentoken.norm.dat" => "data/txt_sentoken.all.dat" do |t|
  sh "./do_norm_poldata.py < #{t.prerequisites.first} > #{t.name}"
end

#
# Create a single "master" file containing all instances.
#
file "data/txt_sentoken.all.dat" => \
  ["data/txt_sentoken/neg", "data/txt_sentoken/pos"] do |t|
  
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
  `ls -1 #{prereqs[0]}/* | parallel --gnu "sed 's/^/0#{sep}{/.}#{sep}/g' {}" >> #{target}`
  `ls -1 #{prereqs[1]}/* | parallel --gnu "sed 's/^/1#{sep}{/.}#{sep}/g' {}" >> #{target}`
end

CLEAN << "data/txt_sentoken.all.dat"

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
