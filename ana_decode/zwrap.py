#

# wrap from plain txt into sgm form
import sys

def main():
    file_src, file_refs = sys.argv[1], sys.argv[2:]
    #
    with open(file_src) as fd:
        lines_src = [line.strip() for line in fd]
    lines_refs = []
    for f_ref in file_refs:
        with open(f_ref) as fd:
            lines_ref = [line.strip() for line in fd]
            lines_refs.append(lines_ref)
    # src
    with open("src.sgm", "w") as fd:
        fd.write('<srcset setid="zset" srclang="any">\n')
        fd.write('<doc docid="zdoc" genre="news" origlang="zh">\n')
        for i, line in enumerate(lines_src):
            fd.write('<seg id="%d"> %s </seg>\n' % (i+1, line))
        fd.write("</doc>\n")
        fd.write("</srcset>")
    # ref
    with open("refs.sgm", "w") as fd:
        fd.write('<refset trglang="en" setid="zset" srclang="any">\n')
        for idx, lines_ref in enumerate(lines_refs):
            fd.write('<doc sysid="ref%d" docid="zdoc" genre="news" origlang="zh">\n' % idx)
            for i, line in enumerate(lines_ref):
                fd.write('<seg id="%d"> %s </seg>\n' % (i+1, line))
            fd.write("</doc>\n")
        fd.write("</refset>")

if __name__ == "__main__":
    main()
