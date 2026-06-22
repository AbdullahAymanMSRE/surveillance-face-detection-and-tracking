import Link from "next/link";
import { listPeople, thumbnailUrl } from "@/lib/api";
import { CornerBrackets } from "@/components/CornerBrackets";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const people = await listPeople();

  return (
    <main className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10">
      <header className="mb-10 flex flex-wrap items-end justify-between gap-4 border-b border-edge pb-6">
        <div>
          <p className="mb-1 text-xs tracking-[0.3em] text-faint">
            FACE-DASH // ENROLLMENT CONSOLE
          </p>
          <h1 className="font-display text-4xl font-bold tracking-wide text-ink">
            ENROLLED IDENTITIES
          </h1>
        </div>
        <Link
          href="/enroll"
          className="border border-green/60 px-5 py-2.5 text-sm uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void"
        >
          + Enroll new person
        </Link>
      </header>

      {people.length === 0 ? (
        <div className="relative border border-dashed border-edge px-6 py-16 text-center">
          <p className="text-sm uppercase tracking-widest text-faint">
            No identities in the database
          </p>
          <p className="mt-2 text-xs text-faint/70">
            Enroll a person to begin tracking matches.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 md:grid-cols-4">
          {people.map((person, i) => (
            <div
              key={person.id}
              className="group relative border border-edge bg-surface p-3 transition-colors hover:border-edge-bright"
            >
              <CornerBrackets color="var(--green)" size={10} />
              <div className="relative mb-3 aspect-square overflow-hidden border border-edge bg-void">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={thumbnailUrl(person)}
                  alt={person.name}
                  className="h-full w-full object-cover grayscale-[15%] contrast-125"
                />
                <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-green shadow-[0_0_6px_var(--green)] rec-dot" />
              </div>
              <p className="truncate text-sm font-medium tracking-wide text-ink">
                {person.name}
              </p>
              <p className="text-[11px] tracking-widest text-faint">
                ID-{String(person.id).padStart(3, "0")} · #{i + 1}
              </p>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
