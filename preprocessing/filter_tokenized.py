import os
import json
from glob import glob
from optparse import OptionParser


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COFEA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--min-length', type=int, default=9,
                      help='Min tokens for inclusion: default=%default')
    parser.add_option('--source', type=str, default=None,
                      help='Source to import [founders|statutes|farrands|elliots|hein|evans] (None=all): default=%default')

    (options, args) = parser.parse_args()

    cofea_dir = options.basedir
    model_name_or_path = options.model
    min_length = options.min_length
    source = options.source

    indir = os.path.join(cofea_dir, 'unfiltered_' + model_name_or_path)
    outdir = os.path.join(cofea_dir, 'tokenized_' + model_name_or_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if source is not None:
        files = [os.path.join(indir, source + '.jsonlist')]
    else:
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    for infile in files:
        basename = os.path.basename(infile)
        print(basename)
        with open(infile) as f:
            lines = f.readlines()
        print(len(lines))

        outlines = []
        for line in lines:
            line = json.loads(line)
            tokens = line['tokens']
            if line['lang'] == '__label__en' and len(tokens) >= min_length: 
                outlines.append(line)
        print(len(outlines))

        outfile = os.path.join(outdir, basename)
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')

    print("Done")        


if __name__ == '__main__':
    main()
