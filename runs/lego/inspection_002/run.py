import hashlib, itertools, json, sys, platform
from pathlib import Path
import cv2, numpy as np, PIL
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description="Repeat dataset and camera checks on the handed-off Lego scene")
parser.add_argument("output", type=Path, help="Path of a NEW results directory")
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=False)
def save(name, obj):
    with (out / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
root = Path(__file__).resolve().parents[3]
scene = root / "data/raw/nerf_synthetic/lego"
ledger = root / "runs/lego/p00-p06_001/SHA256SUMS"
known = dict(line.split("  ",1)[::-1] for line in ledger.read_text().splitlines())
actual = {str(p.relative_to(scene)) for p in scene.rglob("*") if p.is_file()}
assert actual == set(known)
for name, digest in known.items():
    assert hashlib.sha256((scene/name).read_bytes()).hexdigest() == digest, name

frames = []
counts = {}
for split, count in (("train",100),("val",100),("test",200)):
    doc = json.loads((scene/f"transforms_{split}.json").read_text())
    seen = set()
    for frame in doc["frames"]:
        rel = Path(frame["file_path"]+".png")
        assert rel.parent == Path(split) and rel.stem.startswith("r_") and rel.stem[2:].isdigit()
        assert rel not in seen
        seen.add(rel)
        with Image.open(scene/rel) as im:
            im.load()
            assert (im.format,im.size,im.mode)==("PNG",(800,800),"RGBA")
            assert im.getchannel("A").getextrema()==(0,255)
        pose = np.array(frame["transform_matrix"])
        assert pose.shape == (4,4) and np.isfinite(pose).all()
        frames.append(dict(path=str(rel),split=split,size=[800,800],mode="RGBA",alpha=[0,255],
                           sha256=known[str(rel)],angle=doc["camera_angle_x"],pose=pose.tolist()))
    assert seen == {p.relative_to(scene) for p in (scene/split).glob("r_*.png") if p.stem[2:].isdigit()}
    assert len(seen)==count
    counts[split]=count
for kind in ("depth","normal"):
    for i in range(200):
        with Image.open(scene/f"test/r_{i}_{kind}_0001.png") as im:
            im.load()
            assert (im.format,im.size)==("PNG",(800,800))

save("dataset.json", dict(status="pass", checked_sha256=len(known), counts=counts, auxiliary={"depth":200,"normal":200},missing=0,orphan_color=0,duplicate_paths=0))
save("frames.json", frames)
print("Dataset checks passed", flush=True)

def geometry(row):
    pose = np.array(row["pose"])
    R,C = pose[:3,:3],pose[:3,3]
    w,h = row["size"]
    angle = row["angle"]
    assert 0 < angle < np.pi
    f = w/(2*np.tan(angle/2))
    fy_angle = 2*np.arctan(h/(2*f))
    K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
    V = np.linalg.inv(pose @ np.diag([1,-1,-1,1]))
    orth = abs(R.T@R-np.eye(3)).max()
    det = np.linalg.det(R)
    center_error = np.linalg.norm(V[:3,:3]@C+V[:3,3])
    assert abs(pose[3]-[0,0,0,1]).max()<1e-7 and orth<1e-5 and abs(det-1)<1e-5
    assert center_error < 1e-8
    uv = np.array(list(itertools.product((0,w/2,w-1),(0,h/2,h-1))))
    rays = np.column_stack([(uv[:,0]-w/2)/f,-(uv[:,1]-h/2)/f,-np.ones(9)]) @ R.T
    errs=[]
    for depth in (1,4):
        world = C + depth*rays
        xyz = world@V[:3,:3].T+V[:3,3]
        assert (xyz[:,2]>0).all()
        ndc = xyz[:,:2]/xyz[:,2:3]/np.tan([angle/2,fy_angle/2])
        raster = ((ndc+1)*[w,h]-1)/2
        errs.extend(np.linalg.norm(raster-(uv-.5),axis=1))
    origin = V[:3,3]
    offset = np.linalg.norm(f*origin[:2]/origin[2])
    assert max(errs)<1e-6 and origin[2]>0 and offset<0.01
    return dict(path=row["path"],K=K.tolist(),fov_y=fy_angle,C=C.tolist(),forward=(-R[:,2]).tolist(),
                w2c=V.tolist(),ortho=orth,det=det,radius=np.linalg.norm(C),origin_px=offset,
                center_error=center_error,ray_px=max(errs))

def distances(p,q,view1,view2,k):
    relative = view2 @ np.linalg.inv(view1)
    a,b,c = relative[:3,3]
    cross = np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
    ki = np.linalg.inv(k)
    F = ki.T @ cross @ relative[:3,:3] @ ki
    l2,l1 = p@F.T,q@F
    return abs((q*l2).sum(1))/np.sqrt((l2[:,:2]**2+l1[:,:2]**2).sum(1))

def compare(cams, anchor=None):
    centers = np.array([c["C"] for c in cams])
    dirs = centers/np.linalg.norm(centers,axis=1,keepdims=True)
    angles = np.degrees(np.arccos(np.clip(dirs@dirs.T,-1,1)))
    targets = range(len(cams)) if anchor is None else [anchor]
    pairs = set()
    for i in targets:
        eligible = angles[i] >= 2.0
        if anchor is not None:
            eligible &= [Path(c["path"]).parent != Path(cams[i]["path"]).parent for c in cams]
        assert eligible.any()
        j = int(np.argmin(np.where(eligible, abs(angles[i]-8), np.inf)))
        pairs.add(tuple(sorted((i,j))))
    pairs = sorted(pairs)
    detector = cv2.SIFT_create(nfeatures=1200)
    features={}
    for index in sorted({index for pair in pairs for index in pair}):
        c = cams[index]
        with Image.open(scene/c["path"]) as im:
            rgba = np.array(im)
        pts,desc=detector.detectAndCompute(cv2.cvtColor(rgba,cv2.COLOR_RGBA2GRAY),
                                           (rgba[:,:,3]>240).astype("uint8")*255)
        features[index] = (np.array([k.pt for k in pts]),desc)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def good(a,b):
        return {(m.queryIdx,m.trainIdx) for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.75*n.distance}
    K=np.array(cams[0]["K"])
    assert all(np.allclose(c["K"],K) for c in cams)
    K[:2,2]-=.5
    views=np.array([c["w2c"] for c in cams])
    results=[]
    for i,j in pairs:
        a,da=features[i]
        b,db=features[j]
        matches=sorted(good(da,db)&{(y,x) for x,y in good(db,da)})
        ia,ib=np.array(matches).T
        p=np.column_stack([a[ia],np.ones(len(ia))])
        q=np.column_stack([b[ib],np.ones(len(ib))])
        err=distances(p,q,views[i],views[j],K)
        wrong=int(np.where(angles[i]>=2,angles[j],-1).argmax())
        ctrl=distances(p,q,views[i],views[wrong],K)
        keep=err<=1
        pos=0
        if keep.any():
            X=cv2.triangulatePoints(K@views[i,:3],K@views[j,:3],p[keep,:2].T,q[keep,:2].T)
            X/=X[3]
            pos=int((((views[i]@X)[2]>0)&((views[j]@X)[2]>0)).sum())
        results.append(dict(a=cams[i]["path"],b=cams[j]["path"],angle=angles[i,j],n=len(matches),
            near=int(keep.sum()),front=pos,median=np.median(err),control_pose=cams[wrong]["path"],
            ctrl=int((ctrl<=1).sum()),ctrl_median=np.median(ctrl)))
    return results

cams=[geometry(row) for row in frames]
save("cameras.json", cams)
cv2.setNumThreads(2)
pairs=[]
for split in counts:
    pairs.extend(compare([c for c,r in zip(cams,frames) if r["split"]==split]))
    print(split, "pairs measured", flush=True)
poses=np.array([r["pose"] for r in frames])
near_pairs=[]
for i,pose in enumerate(poses):
    diff=abs(poses[:i]-pose).max(axis=(1,2))
    for j in np.flatnonzero(diff<=1e-5):
        near_pairs.append([frames[j]["path"],frames[i]["path"],diff[j]])
save("pairs.json", pairs)
supported = {p[k] for p in pairs if p["front"] >= 10 for k in ("a","b")}
followup = []
for index,camera in enumerate(cams):
    if camera["path"] not in supported:
        followup.extend(compare(cams,index))
save("followup.json",followup)
combined = supported | {p[k] for p in followup if p["front"] >= 10 for k in ("a","b")}
summary = dict(versions=dict(python=platform.python_version(),numpy=np.__version__,pillow=PIL.__version__,opencv=cv2.__version__),
               dataset="pass",geometry="pass", covered_by_ten_points=len(supported),
               unsupported=sorted({r["path"] for r in frames}-supported), near_poses=near_pairs, splits={},
               covered_after_followup=len(combined), unsupported_after_followup=sorted({r["path"] for r in frames}-combined))
for split in counts:
    selected=[p for p in pairs if p["a"].startswith(split+"/")]
    summary["splits"][split]=dict(pairs=len(selected), **{k:sum(p[k] for p in selected) for k in ("n","near","front","ctrl")},
        min_front=min(p["front"] for p in selected), median_pair_error=np.median([p["median"] for p in selected]),
        median_wrong_pose_error=np.median([p["ctrl_median"] for p in selected]))
summary["camera_maxima"]= {k:max(c[k] for c in cams) for k in ("ortho","origin_px","center_error","ray_px")}
summary["camera_maxima"]["det_error"]=max(abs(c["det"]-1) for c in cams)
save("manifest.json", summary)
print(json.dumps(summary, indent=2))

