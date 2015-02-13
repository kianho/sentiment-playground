task :default => ["data/txt_sentoken.all.dat"] do
end

#
# Create a single file containing all instances
#
file "data/txt_sentoken.all.dat" => \
  ["data/txt_sentoken/neg", "data/txt_sentoken/pos"] do |t|
  
  # Prepend the class label to each line ('0' -> negative, '1' -> positive).
  #
  # NOTE: the class label is always the first character in the line, it is
  # followed by a single space. The remaining characters in the line belong
  # to the sentiment sentence.
  #
  # .'. Ignore the first two characters of each line to get only the sentance.
  `ls -1 #{t.prerequisites[0]}/* | sort -n | xargs -I{} cat {} | awk '{print "0 " $0}' > #{t.name}`
  `ls -1 #{t.prerequisites[1]}/* | sort -n | xargs -I{} cat {} | awk '{print "1 " $0}' >> #{t.name}`
end


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
