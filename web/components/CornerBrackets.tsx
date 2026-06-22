type CornerBracketsProps = {
  color?: string;
  size?: number;
};

export function CornerBrackets({ color = "var(--green)", size = 14 }: CornerBracketsProps) {
  const base = "absolute pointer-events-none";
  const style = { width: size, height: size, borderColor: color };
  return (
    <>
      <span className={`${base} top-0 left-0 border-t-2 border-l-2`} style={style} />
      <span className={`${base} top-0 right-0 border-t-2 border-r-2`} style={style} />
      <span className={`${base} bottom-0 left-0 border-b-2 border-l-2`} style={style} />
      <span className={`${base} bottom-0 right-0 border-b-2 border-r-2`} style={style} />
    </>
  );
}
