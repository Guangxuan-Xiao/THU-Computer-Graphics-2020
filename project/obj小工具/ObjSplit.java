import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;
import java.text.NumberFormat;

public class ObjSplit {

	class Vec3 {
		public float x[];

		public Vec3() {
			x = new float[3];
		}

		public Vec3(String[] tok) {
			x = new float[3];

			for (int ii = 0; ii < 3; ii++) {
				x[ii] = Float.parseFloat(tok[ii + 1]);
			}
		}

		public String toString() {
			NumberFormat nf = NumberFormat.getInstance();
			nf.setMaximumFractionDigits(8);
			nf.setGroupingUsed(false);
			return nf.format(x[0]) + " " + nf.format(x[1]) + " " + nf.format(x[2]);
		}

	}

	class Vec2 {
		public float x[];

		public Vec2() {
			x = new float[2];
		}

		public Vec2(String[] tok) {
			x = new float[2];

			for (int ii = 0; ii < 2; ii++) {
				x[ii] = Float.parseFloat(tok[ii + 1]);
			}
		}

		public String toString() {
			NumberFormat nf = NumberFormat.getInstance();
			nf.setMaximumFractionDigits(8);
			nf.setGroupingUsed(false);
			return nf.format(x[0]) + " " + nf.format(x[1]);
		}

	}

	class Int3 {
		public int x[];
		public int tex[];
		public int n[];

		public Int3() {
			x = new int[3];
		}

		public Int3(String[] tok) {
			x = new int[3];
			tex = new int[3];
			n = new int[3];
			for (int ii = 0; ii < 3; ii++) {
				String s[] = tok[1 + ii].split("\\/");
				x[ii] = Integer.parseInt(s[0]);
				if (s.length > 1) {
					tex[ii] = Integer.parseInt(s[1]);
				}
				if (s.length > 2) {
					n[ii] = Integer.parseInt(s[2]);
					// System.out.println("n!");
					// System.out.println(n[ii]);
				}
			}
		}

		public String toString() {
			return x[0] + " " + x[1] + " " + x[2];
		}
	}

	class Material {
		public Vec3 specular;
		public Vec3 diffuse;
		public Vec3 emission;
		public float shininess;
		public float refraction;
		public String texture;
		public String name;
		public String bump;

		public void print(PrintStream ps) {
			if (emission != null) {
				ps.println("emission " + emission);
			}
			if (specular != null) {
				ps.println("specularColor " + specular);
			}
			if (diffuse != null) {
				ps.println("diffuseColor " + diffuse);
			}
			if (shininess > 0) {
				ps.println("shininess " + shininess);
			}
			if (refraction > 0) {
				ps.println("refr " + refraction);
			}
			if (texture != null) {
				ps.println("texture tex/" + texture);
			}
			if (bump != null) {
				ps.println("bump bump/" + texture);
			}
		}

		public Material() {

		}

		public Material(Scanner f) {
			refraction = 0;
			shininess = 0;
			while (f.hasNextLine()) {
				String line = f.nextLine();
				line = line.replaceAll("^\\s+", "");
				if (line.length() < 1) {
					break;
				}
				String toks[] = line.split("\\s+");
				if (toks[0].equals("Ns")) {
					shininess = Float.parseFloat(toks[1]);
				} else if (toks[0].equals("Ni")) {
					refraction = Float.parseFloat(toks[1]);
				} else if (toks[0].equals("Kd")) {
					diffuse = new Vec3(toks);
				} else if (toks[0].equals("Ke")) {
					emission = new Vec3(toks);
				} else if (toks[0].equals("Ks")) {
					specular = new Vec3(toks);
				} else if (toks[0].equals("map_Kd")) {
					texture = toks[1].replaceAll("\\.\\w+", "\\.bmp");
				} else if (toks[0].equals("bump")) {
					bump = toks[1].replaceAll("\\.\\w+", "\\.bmp");
				}
			}
		}
	}

	ArrayList<Vec3> v, vn;
	ArrayList<Vec2> vt;

	ArrayList<ArrayList<Int3>> parts;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println(args[0]);
		System.out.println(args[1]);
		new ObjSplit(args[0], args[1]);
	}

	public ObjSplit(String objfilename, String mtlfilename) {
		Scanner oFile = FileUtil.openInFile(objfilename);
		Scanner mFile = FileUtil.openInFile(mtlfilename);

		// build material index
		// maps from material name to its index
		HashMap<String, Integer> nameMap = new HashMap<String, Integer>();
		HashMap<Integer, String> idxMap = new HashMap<Integer, String>();
		int matIdx = 0;
		ArrayList<Material> matList = new ArrayList<Material>();
		while (mFile.hasNextLine()) {
			String line = mFile.nextLine();
			line = line.replaceAll("^\\s+", "");
			if (line.length() < 1) {
				continue;
			}
			if (line.charAt(0) == '#') {
				continue;
			}
			String toks[] = line.split("\\s+");
			if (toks[0].equals("newmtl")) {
				String matName = toks[1];
				nameMap.put(matName, matIdx);
				idxMap.put(matIdx, matName);
				matIdx++;
				Material mat = new Material(mFile);
				matList.add(mat);
			}
		}
		mFile.close();

		int partIdx = 0;

		parts = new ArrayList<ArrayList<Int3>>();
		for (int ii = 0; ii < matList.size(); ii++) {
			parts.add(new ArrayList<Int3>());
		}
		v = new ArrayList<Vec3>();
		vn = new ArrayList<Vec3>();
		vt = new ArrayList<Vec2>();

		while (oFile.hasNextLine()) {
			String line = oFile.nextLine();
			line = line.replaceAll("^\\s+", "");
			if (line.length() < 1) {
				continue;
			}
			if (line.charAt(0) == '#') {
				continue;
			}
			String toks[] = line.split("\\s+");
			// System.out.println(toks[1]);
			if (toks[0].equals("v")) {
				v.add(new Vec3(toks));
			} else if (toks[0].equals("vt")) {
				vt.add(new Vec2(toks));
			} else if (toks[0].equals("vn")) {
				vn.add(new Vec3(toks));
			} else if (toks[0].equals("f")) {

				if (toks.length == 5) {
					String subtoks1[] = new String[] { toks[2], toks[3], toks[1], toks[4] };
					parts.get(partIdx).add(new Int3(subtoks1));
					String subtoks2[] = new String[] { toks[0], toks[1], toks[3], toks[2] };
					parts.get(partIdx).add(new Int3(subtoks2));
				} else
					parts.get(partIdx).add(new Int3(toks));
			} else if (toks[0].equals("usemtl")) {
				partIdx = nameMap.get(toks[1]);
			}
		}

		oFile.close();

		PrintStream ps = FileUtil.openOutFile("s.txt");
		ps.println("Materials {");
		ps.println("numMaterials " + matList.size());
		for (int ii = 0; ii < matList.size(); ii++) {
			ps.println("Material {");
			matList.get(ii).print(ps);
			ps.println("}");
		}
		ps.println("}");

		ps.println("Group {");
		ps.println("numObjects " + parts.size());
		for (int ii = 0; ii < parts.size(); ii++) {
			ps.println("MaterialIndex " + ii);
			ps.println("TriangleMesh {");
			String filename = "tex/" + idxMap.get(ii) + ".obj";
			printObj(ii, filename);
			ps.println("obj_file " + filename);
			ps.println("}");
		}
		ps.println("}");
		ps.close();

	}

	public void printObj(int idx, String filename) {
		PrintStream ps = FileUtil.openOutFile(filename);
		int[] vidxMap = new int[v.size()];
		int[] vtidxMap = new int[vt.size()];
		int[] vnidxMap = new int[vn.size()];
		int nv = 1, nvt = 1, nvn = 1;
		ArrayList<Int3> flist = parts.get(idx);
		ArrayList<Vec3> newv = new ArrayList<Vec3>();
		ArrayList<Vec3> newvn = new ArrayList<Vec3>();
		ArrayList<Vec2> newvt = new ArrayList<Vec2>();
		// obj is 1 based
		for (int ii = 0; ii < flist.size(); ii++) {
			Int3 trig = flist.get(ii);
			for (int jj = 0; jj < 3; jj++) {
				// 1 based
				int vidx = trig.x[jj] - 1;
				if (vidxMap[vidx] == 0) {
					vidxMap[vidx] = nv;
					newv.add(v.get(vidx));
					nv++;
				}
				int vtidx = trig.tex[jj] - 1;
				if (vtidxMap[vtidx] == 0) {
					vtidxMap[vtidx] = nvt;
					newvt.add(vt.get(vtidx));
					nvt++;
				}
				int vnidx = trig.n[jj] - 1;
				if (vnidxMap[vnidx] == 0) {
					vnidxMap[vnidx] = nvn;
					newvn.add(vn.get(vnidx));
					nvn++;
				}
			}
		}
		for (int ii = 0; ii < newv.size(); ii++) {
			ps.println("v " + newv.get(ii));
		}
		for (int ii = 0; ii < newvn.size(); ii++) {
			ps.println("vn " + newvn.get(ii));
		}
		for (int ii = 0; ii < newvt.size(); ii++) {
			ps.println("vt " + newvt.get(ii));
		}
		for (int ii = 0; ii < flist.size(); ii++) {
			ps.print("f");
			Int3 f = flist.get(ii);
			for (int jj = 0; jj < 3; jj++) {
				int vidx = f.x[jj] - 1;
				int vtidx = f.tex[jj] - 1;
				int vnidx = f.n[jj] - 1;
				vidx = vidxMap[vidx];
				vtidx = vtidxMap[vtidx];
				vnidx = vnidxMap[vnidx];
				ps.print(" " + vidx + "/" + vtidx + "/" + vnidx);
			}
			ps.println();
		}
		ps.close();
	}
}
